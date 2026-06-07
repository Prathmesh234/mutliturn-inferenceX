#!/usr/bin/env python3
"""
Convert merged batch_*.jsonl trace files into a HuggingFace-ready dataset.

Output schema (one row per solved problem):
  kernelbook_uuid : int    provenance id (KernelBook)
  entry_point     : str    reference class name
  pytorch_problem : str    the PyTorch reference module (the task input)
  triton_solution : str    final/best generated Triton kernel
  correct         : bool    (flat, for filtering)
  speedup         : float   (flat, for filtering/sorting)
  num_turns       : int     number of judge evaluations (flat)
  turns           : list<struct>  per-iteration {attempt,kernel,status,correct,speedup,feedback}
  messages        : list<struct>  {role,content} ChatML conversation (for SFT/RL)
  result          : struct  {status,correct,speedup,ref_ms,new_ms,max_abs_err,detail}
  metadata        : struct  {model,effort,claude_status,num_turns,num_evals,cost_usd,
                             session_id,gpu_health,elapsed_s}
  trace           : str     the entire raw stream-json trace (JSON string)
  model           : str     generation model
  source          : str     "GPUMODE/KernelBook"
  repo_name/repo_link/licenses/stars : provenance + LICENSE attribution (from KernelBook)

Run on a compute node (needs the aarch64 uv venv with `datasets`):
  srun --partition=gb300 --gres=gpu:1 --time=00:20:00 bash -lc '
    export PATH=$HOME/.local/uv-$(uname -m):$PATH; cd ~/gpumode-triton
    uv run python to_hf.py runs_full/batch_0.jsonl --out hf/batch_0.parquet'
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path


def _parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def trace_durations(trace):
    for e in reversed(trace or []):
        if e.get("type") == "result":
            return e.get("duration_ms"), e.get("duration_api_ms")
    return None, None

# Handles both `python judge.py <<'EOF'` and `cat <<'EOF' | python judge.py`
# (i.e. arbitrary text after the <<'DELIM' up to end-of-line, then the body).
HEREDOC_RE = re.compile(r"<<\s*'?(\w+)'?[^\n]*\n(.*?)\n\1\b", re.DOTALL)


def _result_text(tool_result):
    c = tool_result.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return "" if c is None else str(c)


def _extract_heredoc(cmd):
    m = HEREDOC_RE.search(cmd or "")
    return m.group(2) if m else None


_SED_INPLACE = re.compile(r"sed\s+-i\s+'s(.)(.*?)\1(.*?)\1(g?)'\s+(\S+)")
_SED_REDIR = re.compile(r"sed\s+'s(.)(.*?)\1(.*?)\1(g?)'\s+(\S+)\s*>\s*(\S+)")
_CP = re.compile(r"^\s*cp\s+(\S+)\s+(\S+)")
_READ_REDIR = re.compile(r"<\s*(\S+\.py)")
_READ_CAT = re.compile(r"cat\s+(\S+\.py)\s*\|")


# `cat > file <<'EOF' ... EOF`  (redirect then heredoc)
_WRITE_REDIR = re.compile(r"(?:cat|tee)\s*>\s*(\S+)\s*<<\s*'?(\w+)'?[^\n]*\n(.*?)\n\2\b", re.DOTALL)
# `cat <<'EOF' > file ... EOF`  (heredoc then redirect)
_WRITE_REDIR2 = re.compile(r"<<\s*'?(\w+)'?\s*>\s*(\S+)[^\n]*\n(.*?)\n\1\b", re.DOTALL)


def _bn(p):
    return p.rsplit("/", 1)[-1] if p else p


def _resolve_kernel(cmd, files):
    """The kernel a judge command feeds to stdin: from a heredoc piped to judge, from
    a `cat > f <<EOF` written file, or from a Wrote/Edit/sed'd file then redirected.
    Files are keyed by BASENAME (Claude writes abs paths but reads relative to cwd)."""
    # `cat > file <<EOF ... EOF` / `cat <<EOF > file ... EOF` writing a file
    for m in _WRITE_REDIR.finditer(cmd):
        files[_bn(m.group(1))] = m.group(3)
    for m in _WRITE_REDIR2.finditer(cmd):
        files[_bn(m.group(2))] = m.group(3)
    # heredoc piped straight to judge
    hd = _extract_heredoc(cmd)
    if hd is not None and "judge.py" in cmd.split(hd)[0]:
        return hd
    if hd is not None and not (_WRITE_REDIR.search(cmd) or _WRITE_REDIR2.search(cmd)):
        return hd
    for stmt in re.split(r"[\n;]|&&", cmd):  # replay file-mutating statements in order
        m = _SED_INPLACE.search(stmt)
        if m:
            _, a, b, g, f = m.groups(); f = _bn(f)
            if f in files:
                files[f] = files[f].replace(a, b) if g else files[f].replace(a, b, 1)
            continue
        m = _SED_REDIR.search(stmt)
        if m:
            _, a, b, g, src, dst = m.groups()
            base = files.get(_bn(src), "")
            files[_bn(dst)] = base.replace(a, b) if g else base.replace(a, b, 1)
            continue
        m = _CP.search(stmt)
        if m and _bn(m.group(1)) in files:
            files[_bn(m.group(2))] = files[_bn(m.group(1))]
    m = _READ_REDIR.search(cmd) or _READ_CAT.search(cmd)
    return files.get(_bn(m.group(1))) if m else None


def extract_judge_calls(trace):
    """Per judge invocation, in order: {kernel, feedback}. Tracks Write/Edit/sed/cat so
    file-redirected submissions (`python judge.py < sol.py`) are recovered too."""
    files, pending, calls = {}, {}, []
    for ev in trace or []:
        t = ev.get("type")
        if t == "assistant":
            for c in (ev.get("message", {}) or {}).get("content", []) or []:
                if c.get("type") != "tool_use":
                    continue
                name, inp = c.get("name"), (c.get("input") or {})
                if name == "Write" and inp.get("file_path"):
                    files[_bn(inp["file_path"])] = inp.get("content", "")
                elif name == "Edit" and _bn(inp.get("file_path")) in files:
                    old, new, fp = inp.get("old_string", ""), inp.get("new_string", ""), _bn(inp["file_path"])
                    files[fp] = files[fp].replace(old, new) if inp.get("replace_all") \
                        else files[fp].replace(old, new, 1)
                elif name == "Bash":
                    cmd = inp.get("command", "") or ""
                    if "judge.py" in cmd:
                        pending[c.get("id")] = _resolve_kernel(cmd, files)
                    else:
                        _resolve_kernel(cmd, files)  # may write files for a later judge call
        elif t == "user":
            ts = ev.get("timestamp")  # tool-result events carry the wall-clock time
            for c in (ev.get("message", {}) or {}).get("content", []) or []:
                if c.get("type") == "tool_result" and c.get("tool_use_id") in pending:
                    calls.append({"kernel": pending.pop(c["tool_use_id"]),
                                  "feedback": _result_text(c), "ended_at": ts})
    for _id, k in pending.items():
        calls.append({"kernel": k, "feedback": "", "ended_at": None})
    return calls


def build_turns(trace, trajectory):
    calls = extract_judge_calls(trace)
    trajectory = trajectory or []
    n = max(len(calls), len(trajectory))
    turns = []
    prev_dt = None
    for i in range(n):
        call = calls[i] if i < len(calls) else {}
        tj = trajectory[i] if i < len(trajectory) else {}
        ended_at = call.get("ended_at")
        dt = _parse_ts(ended_at)
        secs = round((dt - prev_dt).total_seconds(), 3) if (dt and prev_dt) else None
        if dt:
            prev_dt = dt
        turns.append({
            "attempt": tj.get("attempt", i + 1),
            "kernel": call.get("kernel"),
            "status": tj.get("status"),
            "correct": tj.get("correct"),
            "speedup": tj.get("speedup"),
            "feedback": call.get("feedback"),
            "ended_at": ended_at,                 # wall-clock when this eval returned
            "seconds_since_prev": secs,           # time for this turn (None for the first)
        })
    return turns


def build_tool_calls(trace):
    """Chronological per-tool-call timeline: {index, tool, is_judge, ended_at,
    seconds_since_prev}. Every tool-result is timestamped; seconds_since_prev is the
    wall-clock between consecutive tool completions (model thinking + that tool's run)."""
    meta, calls = {}, []
    for e in trace or []:
        t = e.get("type")
        if t == "assistant":
            for c in (e.get("message", {}) or {}).get("content", []) or []:
                if c.get("type") == "tool_use":
                    cmd = (c.get("input") or {}).get("command", "") if c.get("name") == "Bash" else ""
                    meta[c.get("id")] = {"tool": c.get("name"), "is_judge": "judge.py" in (cmd or "")}
        elif t == "user":
            ts = e.get("timestamp")
            for c in (e.get("message", {}) or {}).get("content", []) or []:
                if c.get("type") == "tool_result":
                    m = meta.get(c.get("tool_use_id"), {})
                    calls.append({"tool": m.get("tool"), "is_judge": m.get("is_judge", False),
                                  "ended_at": ts})
    prev = None
    for i, c in enumerate(calls):
        dt = _parse_ts(c["ended_at"])
        c["index"] = i + 1
        c["seconds_since_prev"] = round((dt - prev).total_seconds(), 3) if (dt and prev) else None
        if dt:
            prev = dt
    return calls


def build_messages(trace, system_prompt, task):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    if task:
        msgs.append({"role": "user", "content": task})
    for ev in trace or []:
        t = ev.get("type")
        if t == "assistant":
            parts = []
            for c in (ev.get("message", {}) or {}).get("content", []) or []:
                if c.get("type") == "text":
                    parts.append(c.get("text", ""))
                elif c.get("type") == "tool_use":
                    inp = c.get("input") or {}
                    body = inp.get("command", json.dumps(inp))
                    parts.append(f"<tool:{c.get('name')}>\n{body}\n</tool>")
            if parts:
                msgs.append({"role": "assistant", "content": "\n".join(parts)})
        elif t == "user":
            for c in (ev.get("message", {}) or {}).get("content", []) or []:
                if c.get("type") == "tool_result":
                    msgs.append({"role": "tool", "content": _result_text(c)})
    return msgs


def norm_result(ev):
    ev = ev or {}
    return {k: ev.get(k) for k in
            ("status", "correct", "speedup", "ref_ms", "new_ms", "max_abs_err", "detail")}


def load_kernelbook(name, needed_uuids):
    """Stream KernelBook, keep only the uuids we need -> uuid: {python_code, license...}."""
    from datasets import load_dataset
    needed = set(needed_uuids)
    out = {}
    for r in load_dataset(name, split="train", streaming=True):
        u = str(r.get("uuid"))
        if u in needed and u not in out:
            out[u] = {
                "python_code": r.get("python_code"),
                "entry_point": r.get("entry_point"),
                "repo_name": r.get("repo_name"),
                "repo_link": r.get("repo_link"),
                "licenses": r.get("licenses"),
                "stars": r.get("stars"),
            }
            if len(out) == len(needed):
                break
    return out


def build_task(entry, pytorch_problem):
    return (f"Convert this PyTorch module into an optimized, numerically-equivalent "
            f"Triton implementation named `{entry}New` (real `@triton.jit` kernels).\n\n"
            f"```python\n{pytorch_problem}\n```")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="batch_*.jsonl files")
    ap.add_argument("--out", required=True, help="output .parquet path")
    ap.add_argument("--jsonl", help="also write this .jsonl")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--kernelbook", default="GPUMODE/KernelBook")
    ap.add_argument("--no-kernelbook", action="store_true",
                    help="skip KernelBook re-join (no pytorch_problem/license enrichment)")
    ap.add_argument("--system-prompt", default="system_prompt.md")
    args = ap.parse_args()

    records = []
    for f in args.inputs:
        for line in Path(f).read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"loaded {len(records)} records from {len(args.inputs)} file(s)")

    kb = {}
    if not args.no_kernelbook:
        uuids = [str(r.get("uuid")) for r in records]
        print(f"re-joining KernelBook for {len(set(uuids))} uuids (pytorch_problem + licenses)...")
        kb = load_kernelbook(args.kernelbook, uuids)
        print(f"  matched {len(kb)}/{len(set(uuids))}")

    sp = Path(args.system_prompt)
    system_prompt = sp.read_text() if sp.exists() else ""

    rows = []
    for r in records:
        u = str(r.get("uuid"))
        info = kb.get(u, {})
        pytorch_problem = info.get("python_code")
        entry = r.get("entry_point")
        ev = r.get("eval") or {}
        dur_ms, dur_api_ms = trace_durations(r.get("trace"))
        rows.append({
            "kernelbook_uuid": int(u) if u.isdigit() else None,
            "entry_point": entry,
            "pytorch_problem": pytorch_problem,
            "triton_solution": r.get("generated_code"),
            "correct": bool(ev.get("correct")),
            "speedup": ev.get("speedup"),
            "num_turns": r.get("num_evals"),
            "turns": build_turns(r.get("trace"), r.get("eval_trajectory")),
            "tool_calls": build_tool_calls(r.get("trace")),
            "messages": build_messages(r.get("trace"), system_prompt,
                                       build_task(entry, pytorch_problem) if pytorch_problem else None),
            "result": norm_result(ev),
            "metadata": {
                "model": args.model, "effort": args.effort,
                "claude_status": r.get("claude_status"),
                "num_turns": r.get("num_turns"), "num_evals": r.get("num_evals"),
                "cost_usd": r.get("cost_usd"), "session_id": r.get("session_id"),
                "gpu_health": r.get("gpu_health"), "elapsed_s": r.get("elapsed_s"),
                "duration_ms": dur_ms, "duration_api_ms": dur_api_ms,
            },
            "trace": json.dumps(r.get("trace")),
            "model": args.model,
            "source": "GPUMODE/KernelBook",
            "repo_name": info.get("repo_name"),
            "repo_link": info.get("repo_link"),
            "licenses": info.get("licenses"),
            "stars": info.get("stars"),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    from datasets import Dataset
    ds = Dataset.from_list(rows)
    ds.to_parquet(args.out)
    print(f"wrote {args.out}  ({len(ds)} rows)")
    if args.jsonl:
        with open(args.jsonl, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"wrote {args.jsonl}")

    nc = sum(1 for x in rows if x["correct"])
    nmiss = sum(1 for x in rows if x["pytorch_problem"] is None)
    print(f"summary: rows={len(rows)} correct={nc} "
          f"pytorch_problem_missing={nmiss} columns={list(rows[0].keys()) if rows else []}")


if __name__ == "__main__":
    main()
