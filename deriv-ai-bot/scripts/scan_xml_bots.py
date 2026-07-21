"""Scan xml bots/ folder for contract types and strategy patterns."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "xml bots"


def main() -> None:
    files = sorted(ROOT.glob("*.xml"))
    print(f"files={len(files)} dir={ROOT}")
    contracts: Counter[str] = Counter()
    keywords: Counter[str] = Counter()
    per_file: list[tuple[str, set[str]]] = []

    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        found = set(
            m.upper()
            for m in re.findall(
                r"DIGITOVER|DIGITUNDER|DIGITEVEN|DIGITODD|DIGITMATCH|DIGITDIFF|\bCALL\b|\bPUT\b",
                text,
                re.I,
            )
        )
        for c in found:
            contracts[c] += 1
        per_file.append((f.name, found))

        for m in re.findall(
            r"type=\"([^\"]*(?:trade|purchase|contract|rsi|ema|sma|macd|tick|candle|digit)[^\"]*)\"",
            text,
            re.I,
        ):
            keywords[m] += 1
        for m in re.findall(r"(?:duration_unit|DURATION_UNIT)[^A-Za-z0-9]{0,10}([tms])", text, re.I):
            keywords[f"unit_{m.lower()}"] += 1
        for m in re.findall(r"\b(martingale|RSI|EMA|SMA|MACD|stochastic|oscillator)\b", text, re.I):
            keywords[m.lower()] += 1
        for m in re.findall(r"R_\d+|1HZ\d+V|BOOM\d+|CRASH\d+", text, re.I):
            keywords[f"sym_{m.upper()}"] += 1

    print("\n=== Contract types across files ===")
    for k, v in contracts.most_common():
        print(f"  {k}: {v} files")

    print("\n=== Keywords ===")
    for k, v in keywords.most_common(35):
        print(f"  {k}: {v}")

    print("\n=== Per file contracts ===")
    for name, found in per_file:
        print(f"  {name[:50]:50s} {sorted(found) or ['?']}")


if __name__ == "__main__":
    main()
