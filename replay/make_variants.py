#!/usr/bin/env python3
"""
Build a 60-session replay dataset from the 20 recorded sessions in
replay/batch_1.replay.jsonl by emitting, for each original session, three
versions:

  * v0 -- the ORIGINAL session, byte-identical.
  * v1 -- a refactored, cache-distinct variant.
  * v2 -- a second refactored, cache-distinct variant (distinct from v1 too).

The refactor is a deterministic identifier/number remap applied consistently to
EVERY message's `content`:
  * the KernelBench problem-id number that appears in file paths like
    `.../runs/work/311/reference.py` is shifted by a per-variant offset, so the
    early tokens of messages[1] diverge;
  * the kernel class name (the `entry_point`, e.g. `MultiHeadAttention`, plus its
    `...New` drop-in companion) is renamed to a per-variant, same-length name.

Hard invariants (verified by validate() below):
  * messages[0] (the shared system prompt) stays BYTE-IDENTICAL.
  * message count, roles, and order are preserved.
  * `turns` is copied EXACTLY (prefix_len / max_tokens / delay_before_s / rec_*).
  * session_id / entry_point / kernelbook_uuid are made unique per variant.
  * each remapped message stays within +-15% of the original length.

Nothing here is meant to be runnable Triton code -- it just has to be a
plausible, length-similar, token-distinct refactor so the server prefix cache
does not collapse the 60 sessions back onto the original 20 (beyond the
intentionally-shared system prompt at messages[0]).

Usage:
  python3 replay/make_variants.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "batch_1.replay.jsonl"
DST = HERE / "batch_3x.replay.jsonl"

# Per-variant remap schemes. Each variant gets:
#   - id_offset: integer added to the original kernelbook problem id (these are
#     large + mutually distinct so no variant's ids collide with the originals
#     (311..335) or with each other).
#   - the class renamer (see rename_class) keyed by the variant tag.
# v0 is the untouched original and is not listed here.
VARIANTS = [
    {"tag": "v1", "id_offset": 4000},
    {"tag": "v2", "id_offset": 7000},
]

# Same-length single-character substitution tables, applied to the FIRST
# alphabetic character of the class name so the rename diverges as early as
# possible while exactly preserving length. Different table per variant => v1 and
# v2 produce different names from the same original => mutually cache-distinct.
_FIRST_CHAR_MAP = {
    "v1": str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "QRSTUVWXYZABCDEFGHIJKLMNOP"),
    "v2": str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "HIJKLMNOPQRSTUVWXYZABCDEFG"),
}
# Same-length lowercase tail substitution to push divergence beyond just the
# first char (so e.g. "Net" -> a clearly different 3-char token, not a near-miss).
_TAIL_CHAR_MAP = {
    "v1": str.maketrans("aeiou", "uoiea"),
    "v2": str.maketrans("aeiou", "iouae"),
}


def rename_class(name: str, tag: str) -> str:
    """Return a same-length, variant-specific rename of a class name.

    Length is preserved exactly (character-for-character substitution), which
    keeps `name` and `nameNew` occurrences length-stable everywhere they appear.
    """
    if not name:
        return name
    chars = list(name)
    # Remap the first alphabetic character (early divergence).
    for i, c in enumerate(chars):
        if c.isalpha():
            up = c.upper()
            mapped = up.translate(_FIRST_CHAR_MAP[tag])
            chars[i] = mapped if c.isupper() else mapped.lower()
            break
    # Remap interior vowels for additional divergence (still length-preserving).
    for i, c in enumerate(chars):
        if c.islower():
            chars[i] = c.translate(_TAIL_CHAR_MAP[tag])
        elif c.isupper():
            chars[i] = c.translate({ord(k.upper()): v.upper()
                                    for k, v in
                                    zip("aeiou", "uoiea" if tag == "v1" else "iouae")})
    out = "".join(chars)
    # Guarantee the result actually differs (it always will given the maps, but
    # be defensive for degenerate inputs).
    return out if out != name else (name[:-1] + "Z" if name else name)


def remap_content(text: str, old_id: int, new_id: int,
                  old_cls: str, new_cls: str) -> str:
    """Apply the id + class-name remap to a single message's content string.

    - The numeric problem id is replaced ONLY when it appears as a standalone
      integer (digit boundaries on both sides), so timing numbers like
      `0.0972373335...` that merely contain the id's digits are NOT corrupted.
    - The class name is replaced as a plain substring; verified upstream that
      every occurrence is a clean class-name start whose only trailing
      identifier text is the `New` suffix, so `old_clsNew` -> `new_clsNew`
      happens automatically and consistently.
    """
    # Class rename first (longer, more specific token).
    text = text.replace(old_cls, new_cls)
    # Then the digit-bounded id.
    text = re.sub(rf"(?<!\d){old_id}(?!\d)", str(new_id), text)
    return text


def make_variant(orig: dict, scheme: dict) -> dict:
    tag = scheme["tag"]
    old_id = int(orig["kernelbook_uuid"])
    new_id = old_id + scheme["id_offset"]
    old_cls = orig["entry_point"]
    new_cls = rename_class(old_cls, tag)

    new_messages = []
    for i, m in enumerate(orig["messages"]):
        if i == 0:
            # messages[0] is the shared system prompt -> byte-identical copy.
            new_messages.append({"role": m["role"], "content": m["content"]})
            continue
        new_messages.append({
            "role": m["role"],
            "content": remap_content(m["content"], old_id, new_id,
                                     old_cls, new_cls),
        })

    return {
        "session_id": f"{orig['session_id']}__{tag}",
        "entry_point": f"{new_cls}__{tag}",
        "source": orig["source"],
        "kernelbook_uuid": new_id,
        "messages": new_messages,
        # Copy turns EXACTLY (deep copy of plain dicts) so the load shape is
        # an unchanged 3x scale-up.
        "turns": [dict(t) for t in orig["turns"]],
        "recorded": dict(orig["recorded"]),
    }


def build() -> list[dict]:
    originals = [json.loads(ln) for ln in SRC.read_text().splitlines() if ln.strip()]
    out: list[dict] = []
    for orig in originals:
        out.append(orig)  # v0: byte-identical original
        for scheme in VARIANTS:
            out.append(make_variant(orig, scheme))
    return out, originals


def validate(sessions: list[dict], originals: list[dict]) -> None:
    shared_system = originals[0]["messages"][0]["content"]
    by_id = {s["session_id"]: s for s in sessions}
    orig_by_id = {o["session_id"]: o for o in originals}

    # 1. count
    assert len(sessions) == 60, f"expected 60 sessions, got {len(sessions)}"

    # 4. unique session_ids
    ids = [s["session_id"] for s in sessions]
    assert len(set(ids)) == 60, "duplicate session_id detected"

    msg_counts = []
    prefix_lens = []
    for s in sessions:
        # 3-shape. prefix_len index safety
        n = len(s["messages"])
        msg_counts.append(n)
        for t in s["turns"]:
            prefix_lens.append(t["prefix_len"])
            assert t["prefix_len"] <= n, (
                f"{s['session_id']}: prefix_len {t['prefix_len']} > "
                f"len(messages) {n}")
        # 3. system prompt byte-identical
        assert s["messages"][0]["content"] == shared_system, (
            f"{s['session_id']}: system prompt diverged")
        # JSON round-trips
        json.dumps(s)

    # 5. variants preserve parent turns + roles/counts exactly
    for s in sessions:
        sid = s["session_id"]
        if sid.endswith("__v1") or sid.endswith("__v2"):
            parent = orig_by_id[sid.rsplit("__", 1)[0]]
            assert s["turns"] == parent["turns"], f"{sid}: turns differ from parent"
            assert [m["role"] for m in s["messages"]] == \
                   [m["role"] for m in parent["messages"]], f"{sid}: roles differ"
            assert len(s["messages"]) == len(parent["messages"]), \
                f"{sid}: message count differs"
            # length parity (+-15%) per message, skipping system prompt
            for i in range(1, len(parent["messages"])):
                o = len(parent["messages"][i]["content"])
                nlen = len(s["messages"][i]["content"])
                if o == 0:
                    assert nlen == 0, f"{sid} msg {i}: empty->nonempty"
                    continue
                ratio = nlen / o
                assert 0.85 <= ratio <= 1.15, (
                    f"{sid} msg {i}: length ratio {ratio:.3f} out of +-15%")

    print(f"[validate] sessions: {len(sessions)} (all JSON-valid)")
    print(f"[validate] unique session_ids: {len(set(ids))}/60")
    print(f"[validate] message counts: min={min(msg_counts)} max={max(msg_counts)}")
    print(f"[validate] prefix_len: min={min(prefix_lens)} max={max(prefix_lens)} "
          f"(<= message count always)")
    print("[validate] all 60 system prompts byte-identical to shared prefix: OK")
    print("[validate] variant turns == parent turns, roles/counts match: OK")
    print("[validate] per-message length parity within +-15%: OK")

    # 6. early-divergence spot check on a sample original + its v1/v2
    sample = originals[0]["session_id"]
    base = by_id[sample]["messages"][1]["content"]
    for tag in ("v1", "v2"):
        var = by_id[f"{sample}__{tag}"]["messages"][1]["content"]
        idx = next((k for k in range(min(len(base), len(var)))
                    if base[k] != var[k]), min(len(base), len(var)))
        print(f"[validate] {sample} vs {tag}: messages[1] first differs at "
              f"char index {idx}: orig={base[idx:idx+18]!r} "
              f"{tag}={var[idx:idx+18]!r}")


def main() -> None:
    sessions, originals = build()
    with DST.open("w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[write] wrote {len(sessions)} sessions -> {DST}")
    validate(sessions, originals)


if __name__ == "__main__":
    main()
