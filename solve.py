#!/usr/bin/env python3
"""
KernelBook -> Triton, in parallel across the 4 GPUs of one node.

Design (kept deliberately simple):
  * One worker process per GPU (4 by default).
  * A single shared multiprocessing.Queue of problems. Each worker pulls the
    next problem when it finishes the previous one (dynamic load balancing,
    no static sharding, no central DB).
  * Each worker writes to its OWN gpu_<id>.jsonl (one line per problem), so
    there's no file contention between workers.
  * Resumable: on startup we scan existing gpu_*.jsonl and skip problems that
    are already done.

Per-problem flow inside a worker:
  1. write the PyTorch reference to work/<uuid>/reference.py
  2. run `claude -p` (no turn cap) -> it pipes kernels to judge.py, which saves
     work/<uuid>/submission.py and feeds back correctness+speedup; it must run at
     least --min-evals judge runs, no cap (full trace captured; rate limits waited out)
  3. authoritative re-score of submission.py on this worker's GPU
  4. append one record {uuid, trace, generated_code, eval, eval_trajectory, ...}
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

from claude_runner import run_claude_with_retry

HERE = Path(__file__).resolve().parent
JUDGE = HERE / "judge.py"


def build_prompt(entry, min_evals, ref_path):
    return f"""You are an expert GPU kernel engineer specializing in Triton.

The file `{ref_path}` contains a PyTorch module named `{entry}` (a subclass of
torch.nn.Module), plus `get_init_inputs()` and `get_inputs()` helpers. Read it first.

Write a Triton implementation: a class named `{entry}New` that is a drop-in
replacement for `{entry}`:
  - identical `__init__` signature (so `{entry}New(*args, **kwargs)` accepts the
    same arguments as `{entry}`),
  - identical `forward` signature,
  - identical parameters/buffers, so `load_state_dict` from the reference works
    and outputs are numerically equivalent.
Implement the forward computation with Triton kernels (`@triton.jit`) wherever
it is sensible; trivial glue may stay in PyTorch.

To TEST it, pipe your COMPLETE kernel source on stdin to the judge. Run this
command exactly as written, from wherever you are -- do NOT `cd` anywhere; your
kernel is saved and scored automatically. Use a quoted heredoc, code at column 0:

python {JUDGE} <<'PYEOF'
import torch
... your entire kernel source, defining class {entry}New ...
PYEOF

The judge prints correctness + speedup vs PyTorch and how many iterations you've done.
Iterate: if not correct, fix and pipe again; if correct, make it FASTER and pipe again.

IMPORTANT: each pipe to the judge is ONE evaluation iteration. You are REQUIRED to
run AT LEAST {min_evals} iterations, each exploring a GENUINELY DIFFERENT optimization
(block size, num_warps/num_stages, tiling, kernel fusion, coalesced/vectorized loads,
fp32 accumulation, a different algorithm, ...). Re-piping an identical kernel does NOT
count. There is NO upper limit -- keep going while you can still improve. Thinking and
drafting are free; only running the judge counts. Your BEST correct version is kept
automatically, so experiment freely. Do not stop before the judge confirms you've met
the {min_evals}-iteration minimum; after that, stop only once further optimization
isn't helping."""


def scan_done(outdir):
    done = set()
    for f in Path(outdir).glob("gpu_*.jsonl"):
        try:
            with open(f) as fh:
                for line in fh:
                    try:
                        done.add(str(json.loads(line)["uuid"]))
                    except Exception:
                        pass
        except FileNotFoundError:
            pass
    return done


def load_tasks(args, done):
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    tasks = []
    for row in itertools.islice(ds, args.start, args.start + args.batch_size):
        uuid = str(row.get("uuid"))
        if uuid in done:
            continue
        tasks.append({
            "uuid": uuid,
            "entry_point": row["entry_point"],
            "python_code": row["python_code"],
        })
    return tasks


def gpu_healthy(gpu_id, timeout=30):
    """Tiny CUDA op in a fresh process: is this GPU usable right now?"""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    code = ("import torch; assert torch.cuda.is_available(); "
            "x=torch.randn(4096, device='cuda'); "
            "assert torch.isfinite((x*2).sum()).item(); torch.cuda.synchronize()")
    try:
        p = subprocess.run([sys.executable, "-c", code], env=env,
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def evaluate(workdir, entry, gpu_id, args):
    """Authoritative final score: re-run the judge on the saved submission.py."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TRITON_CACHE_DIR"] = str(workdir / ".triton_cache")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["JUDGE_REF"] = str(workdir / "reference.py")
    env["JUDGE_WORKDIR"] = str(workdir)
    env["JUDGE_ENTRY"] = entry
    if args.fresh_compile:
        env["JUDGE_FRESH_COMPILE"] = "1"
    cmd = [sys.executable, str(JUDGE), "--score"]
    try:
        p = subprocess.run(cmd, cwd=str(workdir), env=env, capture_output=True,
                           text=True, timeout=args.eval_timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    for line in reversed(p.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except Exception:
            continue
    return {"status": "eval_error", "stderr": (p.stderr or "")[-1000:]}


def worker(worker_id, gpu_id, q, args):
    out_path = Path(args.output_dir) / f"gpu_{worker_id}.jsonl"
    workroot = Path(args.output_dir) / "work"

    def log(msg):
        print(f"[w{worker_id}/gpu{gpu_id}] {msg}", flush=True)

    while True:
        task = q.get()
        if task is None:
            log("queue drained, exiting")
            return

        uuid = task["uuid"]
        entry = task["entry_point"]
        workdir = workroot / uuid
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "reference.py").write_text(task["python_code"])
        for stale in ("submission.py", "best.py", "_evalcount",
                      "_evallog.jsonl", "_best_speedup"):
            (workdir / stale).unlink(missing_ok=True)  # fresh start for this problem

        # Pin this worker's GPU + judge config onto the claude process. JUDGE_WORKDIR
        # is where the judge writes ALL artifacts (so it's robust to Claude `cd`-ing
        # anywhere, and workers never share files).
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["TRITON_CACHE_DIR"] = str(workdir / ".triton_cache")
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["JUDGE_REF"] = str(workdir / "reference.py")
        env["JUDGE_WORKDIR"] = str(workdir)
        env["JUDGE_ENTRY"] = entry
        env["JUDGE_MIN_EVALS"] = str(args.min_evals)
        env["JUDGE_MAX_EVALS"] = str(args.max_evals)
        if args.fresh_compile:
            env["JUDGE_FRESH_COMPILE"] = "1"

        log(f"solving {uuid} ({entry})")
        t0 = time.time()
        cres = run_claude_with_retry(
            build_prompt(entry, args.min_evals, str(workdir / "reference.py")),
            workdir, args, log, env)

        # Claude's own optimization trajectory (one entry per judge run).
        trajectory = []
        log_path = workdir / "_evallog.jsonl"
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                try:
                    trajectory.append(json.loads(line))
                except Exception:
                    pass

        # Prefer the best correct kernel (best.py) over whatever was last piped.
        best_path = workdir / "best.py"
        final_path = best_path if best_path.exists() else workdir / "submission.py"
        gen_code = final_path.read_text() if final_path.exists() else None
        if gen_code:
            eval_res = evaluate(workdir, entry, gpu_id, args)  # authoritative final score
        else:
            eval_res = {"status": "no_solution_file"}

        # GPU health: if this eval failed in a GPU-suspicious way, make sure the
        # device is still usable before moving on, so one bad kernel can't cascade.
        gpu_health = "ok"
        if eval_res.get("status") in ("timeout", "runtime_error", "eval_error", "judge_crash"):
            if not gpu_healthy(gpu_id):
                log(f"GPU looks unhealthy after eval={eval_res.get('status')}; "
                    f"cooling down {args.gpu_cooldown}s and re-checking")
                time.sleep(args.gpu_cooldown)
                if gpu_healthy(gpu_id):
                    gpu_health = "recovered"
                else:
                    gpu_health = "unhealthy"
                    log(f"GPU STILL unhealthy (cannot hardware-reset without root); "
                        f"cooling down {4 * args.gpu_cooldown}s")
                    time.sleep(4 * args.gpu_cooldown)

        record = {
            "uuid": uuid,
            "entry_point": entry,
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "claude_status": cres.status,
            "num_turns": cres.meta.get("num_turns"),
            "num_evals": len(trajectory),
            "cost_usd": cres.meta.get("total_cost_usd"),
            "session_id": cres.meta.get("session_id"),
            "elapsed_s": round(time.time() - t0, 1),
            "eval": eval_res,
            "used_best": best_path.exists(),  # True if best.py (best correct) was scored
            "gpu_health": gpu_health,
            "eval_trajectory": trajectory,
            "generated_code": gen_code,
            "trace": cres.events,  # full stream-json trace
        }
        with open(out_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()
        log(f"saved {uuid}: claude={cres.status} eval={eval_res.get('status')} "
            f"speedup={eval_res.get('speedup')} evals={record['num_evals']} "
            f"{record['elapsed_s']}s")


def parse_args():
    ap = argparse.ArgumentParser()
    # batch / dataset
    ap.add_argument("--dataset", default="GPUMODE/KernelBook")
    ap.add_argument("--split", default="train")
    ap.add_argument("--start", type=int, default=0, help="dataset offset")
    ap.add_argument("--batch-size", type=int, default=300)
    # parallelism
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-gpus", type=int, default=4)
    ap.add_argument("--output-dir", default="runs")
    # claude
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--effort", default="medium",
                    help="reasoning effort level for the model (e.g. low/medium/high)")
    ap.add_argument("--min-evals", type=int, default=8,
                    help="REQUIRED floor: Claude must run at least this many judge "
                         "iterations (each a distinct optimization) before it may stop")
    ap.add_argument("--max-evals", type=int, default=0,
                    help="optional ceiling on judge runs per problem; <=0 means no cap "
                         "(the wall-clock --claude-timeout is the backstop)")
    ap.add_argument("--claude-timeout", type=int, default=1800,
                    help="hard wall-clock backstop (s) for a single claude -p call "
                         "(there is no --max-turns cap anymore)")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="retries for non-rate-limit errors")
    ap.add_argument("--err-backoff", type=float, default=15)
    # rate-limit waiting
    ap.add_argument("--rl-base", type=float, default=60,
                    help="base backoff (s) when rate limited")
    ap.add_argument("--rl-max", type=float, default=1800,
                    help="max backoff (s) per rate-limit retry")
    ap.add_argument("--rl-reset-cap", type=float, default=7200,
                    help="cap (s) when a reset time is parsed from the message")
    ap.add_argument("--rl-jitter", type=float, default=20)
    ap.add_argument("--rl-max-attempts", type=int, default=30,
                    help="give up a problem after this many rate-limit waits")
    # eval
    ap.add_argument("--eval-timeout", type=int, default=300,
                    help="hard timeout (s) for evaluating one kernel")
    ap.add_argument("--fresh-compile", action="store_true",
                    help="wipe the Triton compile cache before every eval (forces "
                         "recompilation; does NOT change results, only adds time)")
    ap.add_argument("--gpu-cooldown", type=int, default=30,
                    help="seconds to wait + re-check when the GPU looks unhealthy "
                         "after a failed eval (breaks cascading failures)")
    return ap.parse_args()


def main():
    args = parse_args()
    # Absolute so paths stay correct when the judge runs with cwd=workdir
    # (a relative output_dir would otherwise double: workdir/workdir/reference.py).
    args.output_dir = str(Path(args.output_dir).resolve())
    # Durable context (hardware + triton idioms + harness contract) for the model.
    sp = HERE / "system_prompt.md"
    args.system_prompt = sp.read_text() if sp.exists() else ""
    (Path(args.output_dir) / "work").mkdir(parents=True, exist_ok=True)

    done = scan_done(args.output_dir)
    tasks = load_tasks(args, done)
    print(f"loaded {len(tasks)} tasks "
          f"(offset {args.start}, batch {args.batch_size}, skipped {len(done)} done)",
          flush=True)
    if not tasks:
        print("nothing to do.")
        return

    q = mp.Queue()
    procs = []
    for i in range(args.num_workers):
        gpu = i % args.num_gpus
        p = mp.Process(target=worker, args=(i, gpu, q, args), daemon=False)
        p.start()
        procs.append(p)

    for t in tasks:
        q.put(t)
    for _ in range(args.num_workers):
        q.put(None)  # one sentinel per worker

    for p in procs:
        p.join()
    print("all workers finished.", flush=True)


if __name__ == "__main__":
    main()
