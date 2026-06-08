#!/usr/bin/env python3
"""
Single drop-in judge. Claude pipes its COMPLETE kernel source on stdin; this
runs it on the GPU against the reference and prints correctness + speedup vs
PyTorch. Each invocation is ONE evaluation iteration (the budget).

  Claude (loop):  python judge.py            # reads kernel from stdin
  Worker (final): python judge.py --score    # re-scores saved submission.py -> JSON

Config via env (set by the worker on the `claude -p` process):
  JUDGE_REF        path to reference.py (the PyTorch module + get_inputs helpers)
  JUDGE_ENTRY      reference class name; solution must define <ENTRY>New
  JUDGE_MIN_EVALS  required floor of iterations before Claude may stop (Claude mode)
  JUDGE_MAX_EVALS  optional ceiling; <=0 = no cap (Claude mode only)
CUDA_VISIBLE_DEVICES is pinned to this worker's GPU. The received kernel is saved
to submission.py and every attempt is appended to _evallog.jsonl.
"""
import importlib.util
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

REF = os.environ.get("JUDGE_REF", "reference.py")
ENTRY = os.environ.get("JUDGE_ENTRY", "")
MIN_EVALS = int(os.environ.get("JUDGE_MIN_EVALS", "8"))   # required floor of iterations
MAX_EVALS = int(os.environ.get("JUDGE_MAX_EVALS", "0"))   # <=0 means no cap (wall-clock backstops)
# All artifacts go HERE, not in cwd -- Claude may `cd` elsewhere before running us,
# and multiple workers must never share files. JUDGE_WORKDIR is the per-problem dir.
WORKDIR = Path(os.environ.get("JUDGE_WORKDIR", ".")).resolve()


def _maybe_fresh_compile():
    """Opt-in (--fresh-compile): wipe the Triton compile cache so this eval
    recompiles from scratch. Does NOT affect results, only forces a cold compile."""
    if os.environ.get("JUDGE_FRESH_COMPILE") == "1":
        cache = os.environ.get("TRITON_CACHE_DIR")
        if cache:
            shutil.rmtree(cache, ignore_errors=True)


def _load_module(path, name):
    sys.modules.pop(name, None)  # fresh import so edits take effect
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_init_inputs(gi):
    if isinstance(gi, (list, tuple)) and len(gi) == 2 and isinstance(gi[1], dict):
        return list(gi[0]), dict(gi[1])
    if isinstance(gi, (list, tuple)):
        return list(gi), {}
    return [], {}


def run_eval(sol_path, atol=1e-2, rtol=1e-2):
    """Returns a result dict; never raises."""
    _maybe_fresh_compile()

    # Reward-hacking gate (cheap, static) BEFORE touching the GPU. Rejects the
    # obvious cheats (torch-only, unlaunched kernel, delegation, timing tricks).
    try:
        import reward_hacking
        ok, reasons = reward_hacking.check(Path(sol_path).read_text(), ENTRY)
        if not ok:
            return {"status": "reward_hack", "correct": False,
                    "reasons": reasons, "detail": "; ".join(reasons)}
    except Exception:
        pass  # detector must never block a legit eval

    try:
        import torch
    except Exception as e:
        return {"status": "env_unavailable", "detail": f"torch import failed: {e}"}
    if not torch.cuda.is_available():
        return {"status": "env_unavailable", "detail": "torch.cuda.is_available() is False"}

    import torch.nn as nn
    torch.manual_seed(0)
    dev = "cuda"

    try:
        ref = _load_module(REF, "ref_mod")
        RefClass = getattr(ref, ENTRY)
        ia, ik = _parse_init_inputs(ref.get_init_inputs())
        inputs = [x.to(dev) if hasattr(x, "to") else x for x in ref.get_inputs()]
        ref_model = RefClass(*ia, **ik).to(dev).eval()
    except Exception:
        return {"status": "reference_error", "detail": traceback.format_exc()[-1500:]}

    try:
        sol = _load_module(sol_path, "sol_mod")
    except Exception:
        return {"status": "compile_error", "detail": traceback.format_exc()[-1500:]}

    NewClass = getattr(sol, ENTRY + "New", None)
    if NewClass is None:
        cands = [v for v in vars(sol).values()
                 if isinstance(v, type) and issubclass(v, nn.Module)
                 and getattr(v, "__module__", "") == "sol_mod"]
        NewClass = cands[0] if cands else None
    if NewClass is None:
        return {"status": "no_model_class", "detail": f"no class {ENTRY}New in submission"}

    try:
        new_model = NewClass(*ia, **ik).to(dev).eval()
    except Exception:
        return {"status": "compile_error", "detail": traceback.format_exc()[-1500:]}

    try:
        new_model.load_state_dict(ref_model.state_dict())
    except Exception:
        pass  # best effort

    try:
        with torch.no_grad():
            ref_out = ref_model(*inputs)
            new_out = new_model(*inputs)
    except Exception:
        return {"status": "runtime_error", "detail": traceback.format_exc()[-1500:]}

    def flat(o):
        if isinstance(o, (list, tuple)):
            return torch.cat([t.reshape(-1).float() for t in o])
        return o.reshape(-1).float()

    try:
        r, n = flat(ref_out), flat(new_out)
        if r.shape != n.shape:
            return {"status": "incorrect", "correct": False,
                    "detail": f"shape mismatch {tuple(r.shape)} vs {tuple(n.shape)}"}
        max_abs = (r - n).abs().max().item()
        correct = bool(torch.allclose(r, n, atol=atol, rtol=rtol))
    except Exception:
        return {"status": "runtime_error", "detail": traceback.format_exc()[-1500:]}

    def bench(m):
        try:
            import triton
            return float(triton.testing.do_bench(lambda: m(*inputs)))
        except Exception:
            import time as _t
            with torch.no_grad():
                for _ in range(10):
                    m(*inputs)
                torch.cuda.synchronize()
                s = _t.time()
                for _ in range(50):
                    m(*inputs)
                torch.cuda.synchronize()
            return (_t.time() - s) * 1000.0 / 50.0

    ref_ms = new_ms = speedup = None
    try:
        with torch.no_grad():
            ref_ms = bench(ref_model)
            new_ms = bench(new_model)
        speedup = ref_ms / new_ms if new_ms and new_ms > 0 else None
    except Exception:
        pass

    return {"status": "ok" if correct else "incorrect", "correct": correct,
            "max_abs_err": max_abs, "ref_ms": ref_ms, "new_ms": new_ms,
            "speedup": speedup}


def main():
    # Worker mode: re-score the best (or latest) saved submission, print JSON only.
    if "--score" in sys.argv:
        best = WORKDIR / "best.py"
        sub = best if best.exists() else WORKDIR / "submission.py"
        if not sub.exists():
            print(json.dumps({"status": "no_solution_file"}))
            return
        print(json.dumps(run_eval(str(sub))))
        return

    # Claude mode: read kernel from stdin -> save -> eval -> count -> advise.
    code = sys.stdin.read()
    if not code.strip():
        print("ERROR: no kernel received on stdin. Pipe your COMPLETE kernel source in.")
        return

    cf = WORKDIR / "_evalcount"
    prev = int(cf.read_text()) if cf.exists() else 0

    # Optional hard ceiling (MAX_EVALS<=0 means no cap; wall-clock backstops instead).
    if MAX_EVALS > 0 and prev >= MAX_EVALS:
        print(f"=== eval refused: cap {prev}/{MAX_EVALS} reached ===")
        print(">>> No evaluations remain. Submit your best kernel and STOP now.")
        return

    # Substantive-iteration guard: an identical re-pipe does NOT count toward the
    # minimum. Compare to the previous kernel BEFORE overwriting it.
    sub_path = WORKDIR / "submission.py"
    prev_code = sub_path.read_text() if sub_path.exists() else None
    if prev > 0 and prev_code is not None and code.strip() == prev_code.strip():
        print(f"=== identical submission: NOT counted ({prev}/{MIN_EVALS} so far) ===")
        print(">>> This kernel is byte-identical to your previous attempt, so it does not")
        print(">>> count as an iteration. Change something real (block size, num_warps/")
        print(">>> num_stages, tiling, fusion, vectorized loads, fp32 accumulation, or the")
        print(">>> algorithm) and pipe again.")
        return
    sub_path.write_text(code)  # capture the latest kernel

    nrun = prev + 1
    cf.write_text(str(nrun))

    res = run_eval(str(sub_path))
    res["attempt"] = nrun
    with open(WORKDIR / "_evallog.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")

    # Keep the BEST correct kernel so a later regression can't lose it.
    if res.get("correct") and res.get("speedup") is not None:
        bf = WORKDIR / "_best_speedup"
        prev_best = float(bf.read_text()) if bf.exists() else -1.0
        if res["speedup"] > prev_best:
            shutil.copyfile(sub_path, WORKDIR / "best.py")
            bf.write_text(str(res["speedup"]))

    cap_str = str(MAX_EVALS) if MAX_EVALS > 0 else "inf"
    print(f"=== eval attempt {nrun}  (minimum {MIN_EVALS}, cap {cap_str}) ===")
    print(f"status      : {res.get('status')}")
    print(f"correct     : {res.get('correct')}")
    if res.get("max_abs_err") is not None:
        print(f"max_abs_err : {res['max_abs_err']:.3e}")
    if res.get("speedup") is not None:
        print(f"speedup     : {res['speedup']:.3f}x   (ref {res.get('ref_ms')} ms, new {res.get('new_ms')} ms)")
    if res.get("detail"):
        print(f"detail      : {res['detail'][:800]}")
    print()

    below_floor = nrun < MIN_EVALS
    need = MIN_EVALS - nrun
    at_cap = MAX_EVALS > 0 and nrun >= MAX_EVALS

    if res.get("status") == "reward_hack":
        print(">>> REJECTED — this looks like reward hacking, NOT a valid solution:")
        for r in res.get("reasons", []):
            print(f"      - {r}")
        print(">>> You MUST implement the computation with a REAL Triton kernel:")
        print(">>>   * Do NOT compute the result with high-level PyTorch ops (torch.nn,")
        print(">>>     F.*, torch.matmul, torch.sum, ...). torch may ONLY allocate the")
        print(">>>     output, reshape/contiguous, and launch the kernel.")
        print(">>>   * Your @triton.jit kernel must be LAUNCHED and must tl.load the")
        print(">>>     inputs and tl.store the outputs (do the real math in Triton).")
        print(">>>   * Do NOT import/call the reference, hardcode outputs, cache results,")
        print(">>>     use extra CUDA streams/threads/graphs, or modify the harness.")
        print(">>> Rewrite with a genuine Triton kernel and pipe again.")
    elif res.get("status") == "env_unavailable":
        print(">>> GPU/torch unavailable; cannot evaluate. Stop here.")
    elif below_floor:
        if res.get("correct"):
            print(f">>> Correct — but you've completed only {nrun}/{MIN_EVALS} required "
                  f"iterations. Do NOT stop.")
            print(">>> Try a GENUINELY DIFFERENT optimization (block size, num_warps/")
            print(">>>  num_stages, tiling, kernel fusion, coalesced/vectorized loads, fp32")
            print(f">>>  accumulation, or a different algorithm) and pipe again. {need} more required.")
        else:
            print(f">>> Not correct yet ({nrun}/{MIN_EVALS} required). Fix the kernel and pipe it again.")
    elif at_cap:
        print(">>> Evaluation cap reached. Submit your best kernel and STOP now.")
    elif res.get("correct"):
        print(f">>> Correct! You've met the {MIN_EVALS}-iteration minimum. Keep pushing for "
              f"more speedup with a different approach, or stop if it's truly plateaued.")
    else:
        print(">>> Not correct yet. Fix the kernel and pipe it again.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"status": "judge_crash", "detail": traceback.format_exc()[-1500:]}))
