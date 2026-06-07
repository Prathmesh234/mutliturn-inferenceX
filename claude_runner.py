"""
Thin wrapper around `claude -p` (Claude Code in non-interactive print mode).

Responsibilities:
  * run claude with NO turn cap (the budget is judge.py eval runs), capturing the
    FULL trace (stream-json)
  * detect rate-limit failures and WAIT them out, then retry the same problem
  * retry a small number of times on other transient errors

`claude -p` talks to the Anthropic API over the network; it does NOT use the
local GPU. The GPU is only used by judge.py, to run the kernel.
"""
import json
import random
import re
import subprocess
import time
from dataclasses import dataclass, field

# Signals that a failure was caused by hitting a rate / usage limit.
RATE_PATTERNS = re.compile(
    r"(rate.?limit|usage limit|limit reached|429|overloaded|"
    r"resets?\s+at|too many requests|capacity|quota)",
    re.IGNORECASE,
)
# A unix timestamp (~2023-2033) that some limit messages include as a reset time.
EPOCH_RE = re.compile(r"\b(1[7-9]\d{8}|20\d{8})\b")


@dataclass
class ClaudeResult:
    status: str                       # "ok" | "rate_limited" | "error" | "timeout"
    events: list = field(default_factory=list)   # full parsed stream-json trace
    meta: dict = field(default_factory=dict)      # cost, num_turns, session_id, ...
    error: str = ""
    reset_wait: float | None = None   # seconds to wait, if parsed from a limit msg


def run_claude(prompt, workdir, args, timeout, env):
    """One invocation. Returns a ClaudeResult; never raises for normal failures.

    No --max-turns: file ops are unlimited; the iteration budget is enforced by
    judge.py (eval runs). `timeout` is the hard wall-clock backstop. `env` pins
    CUDA_VISIBLE_DEVICES + judge config so Claude's Bash lands on this GPU.
    """
    cmd = [
        env.get("CLAUDE_BIN", "claude"), "-p", prompt,
        "--model", args.model,
        "--effort", args.effort,
        "--output-format", "stream-json", "--verbose",
        "--dangerously-skip-permissions",
        "--allowedTools", "Read,Write,Edit,Bash",
    ]
    sysprompt = getattr(args, "system_prompt", "")
    if sysprompt:
        cmd += ["--append-system-prompt", sysprompt]
    try:
        p = subprocess.run(
            cmd, cwd=str(workdir), env=env,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ClaudeResult(status="timeout", error=f"claude -p exceeded {timeout}s")
    except FileNotFoundError:
        return ClaudeResult(status="error", error="`claude` binary not found on PATH")

    # stream-json emits one JSON object per line (JSONL) -> the whole trace.
    events = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    result_evt = next(
        (e for e in reversed(events) if e.get("type") == "result"), None
    )
    meta = {}
    if result_evt:
        for k in ("subtype", "is_error", "total_cost_usd", "num_turns",
                  "duration_ms", "duration_api_ms", "session_id", "result"):
            if k in result_evt:
                meta[k] = result_evt[k]

    is_err = (
        p.returncode != 0
        or result_evt is None
        or bool(meta.get("is_error"))
    )

    # Build a haystack of everything that might mention a rate limit.
    haystack = " ".join([
        p.stderr or "",
        json.dumps(result_evt) if result_evt else "",
    ])

    if is_err and RATE_PATTERNS.search(haystack):
        return ClaudeResult(
            status="rate_limited", events=events, meta=meta,
            error=haystack[-500:], reset_wait=_parse_reset(haystack),
        )
    if is_err:
        detail = (p.stderr or json.dumps(meta))[-500:]
        return ClaudeResult(status="error", events=events, meta=meta, error=detail)
    return ClaudeResult(status="ok", events=events, meta=meta)


def _parse_reset(text):
    """If a limit message contains a future unix timestamp, return seconds until it."""
    now = time.time()
    for m in EPOCH_RE.findall(text):
        ts = int(m)
        if now < ts < now + 12 * 3600:      # sanity: within next 12h
            return ts - now
    return None


def run_claude_with_retry(prompt, workdir, args, log, env):
    """
    Run claude, waiting out rate limits and retrying transient errors.

    Rate limits -> sleep (until the parsed reset time, else exponential backoff
    with jitter) and retry the SAME problem. Other errors -> a few quick retries.
    """
    rl_attempts = 0
    err_attempts = 0
    while True:
        res = run_claude(prompt, workdir, args, args.claude_timeout, env)

        if res.status == "ok":
            return res

        if res.status == "rate_limited":
            rl_attempts += 1
            if res.reset_wait:
                wait = min(res.reset_wait, args.rl_reset_cap)
            else:
                wait = min(args.rl_base * (2 ** min(rl_attempts - 1, 5)), args.rl_max)
            wait += random.uniform(0, args.rl_jitter)
            log(f"RATE LIMITED (attempt {rl_attempts}) -> sleeping {int(wait)}s")
            if rl_attempts > args.rl_max_attempts:
                log("giving up after too many rate-limit retries")
                return res
            time.sleep(wait)
            continue

        # timeout / error
        err_attempts += 1
        if err_attempts > args.max_retries:
            log(f"giving up after {err_attempts} errors: {res.error[:200]}")
            return res
        log(f"{res.status} (retry {err_attempts}/{args.max_retries}) "
            f"in {args.err_backoff}s :: {res.error[:160]}")
        time.sleep(args.err_backoff)
        continue
