"""
Convert persona-conditioned SFT JSON files to the simplified
{system, news, position} format.

Sources (read-only):
  persona_sft_training_data_rule.json
  persona_sft_training_data_gpt4.1mini.json

Outputs:
  sft-rule.json
  sft-gpt4-1mini.json

Only the `messages` field of each sample is consulted; every other top-level
field (including the cached `label`) is ignored as required.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parent

INPUT_FILES = [
    (SFT_DIR / "persona_sft_training_data_rule.json", SFT_DIR / "sft-rule.json"),
    (
        SFT_DIR / "persona_sft_training_data_gpt4.1mini.json",
        SFT_DIR / "sft-gpt4-1mini.json",
    ),
]

ALLOWED = {"Long", "Short", "Flat"}

# "News digest:" up to (but not including) the next field marker that the
# original builder appends: "Filing digest:" or the trailing "Decide a single
# position" instruction. DOTALL so the body can span multiple lines.
_NEWS_RE = re.compile(
    r"News digest:\s*(.*?)\s*(?:\n\s*Filing digest:|\n\s*Decide a single position|\Z)",
    re.DOTALL,
)


def _clean(text: str) -> str:
    """Collapse stray escape artifacts and trim. Preserves UTF-8 content."""
    if text is None:
        return ""
    # Replace literal "\n" sequences that may have leaked in from JSON strings.
    text = text.replace("\\n", "\n").replace("\\r", "")
    # Normalise CRLF and trim trailing whitespace per line.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    # Drop leading/trailing blank lines but keep interior structure.
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def extract_news(user_msg: str) -> str:
    m = _NEWS_RE.search(user_msg or "")
    if not m:
        raise ValueError("'News digest:' section not found")
    return _clean(m.group(1))


def extract_position(assistant_msg: str) -> str:
    """Parse the assistant content as JSON and pull 'position'."""
    if assistant_msg is None:
        raise ValueError("assistant content is None")
    raw = assistant_msg.strip()
    obj = json.loads(raw)  # strict JSON parse
    if not isinstance(obj, dict) or "position" not in obj:
        raise ValueError(f"missing 'position' in assistant JSON: {raw!r}")
    pos = str(obj["position"]).strip().capitalize()
    if pos not in ALLOWED:
        raise ValueError(f"disallowed position {pos!r}")
    return pos


def convert_file(src: Path, dst: Path) -> dict:
    with src.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    converted: list[dict] = []
    failures: list[tuple[int, str]] = []
    label_counter: Counter[str] = Counter()

    for idx, sample in enumerate(data):
        try:
            msgs = sample["messages"]
            system = _clean(msgs[0]["content"])
            news = extract_news(msgs[1]["content"])
            position = extract_position(msgs[2]["content"])
        except Exception as exc:
            failures.append((idx, str(exc)))
            print(
                f"  [warn] {src.name}: sample idx={idx} failed -> {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue

        converted.append({"system": system, "news": news, "position": position})
        label_counter[position] += 1

    with dst.open("w", encoding="utf-8") as fh:
        json.dump(converted, fh, ensure_ascii=False, indent=2)

    return {
        "src": src,
        "dst": dst,
        "input_count": len(data),
        "output_count": len(converted),
        "labels": label_counter,
        "failures": failures,
    }


def main() -> None:
    summaries = [convert_file(src, dst) for src, dst in INPUT_FILES]

    print()
    print("=" * 64)
    print("SIMPLIFIED SFT CONVERSION SUMMARY")
    print("=" * 64)
    for s in summaries:
        print(f"\nInput : {s['src']}")
        print(f"Output: {s['dst']}")
        print(f"  total input samples   : {s['input_count']}")
        print(f"  total output samples  : {s['output_count']}")
        print(f"  parsing failures      : {len(s['failures'])}")
        print(f"  label distribution:")
        total = max(sum(s["labels"].values()), 1)
        for lbl in ("Long", "Short", "Flat"):
            c = s["labels"].get(lbl, 0)
            print(f"    {lbl:<6} {c:>5}  ({c / total:.1%})")
        # Validate
        out_set = set(s["labels"].keys())
        valid = out_set.issubset(ALLOWED)
        print(f"  all positions in {{Long,Short,Flat}}: {valid}")
        if s["failures"]:
            print(f"  failed indices (first 10): "
                  f"{[i for i, _ in s['failures'][:10]]}")
    print("=" * 64)


if __name__ == "__main__":
    main()
