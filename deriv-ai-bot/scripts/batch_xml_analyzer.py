"""
Automated Fast Parallel Batch XML Strategy Analyzer

Uses ProcessPoolExecutor to scan all 4,065 XML files in parallel.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
XML_DIR = ROOT / "xml bots"
OUTPUT_REPORT = ROOT / "data" / "xml_batch_analysis.json"


def analyze_single_xml(filepath_str: str) -> Dict[str, Any]:
    filepath = Path(filepath_str)
    rel_name = filepath.name
    res: Dict[str, Any] = {
        "filename": rel_name,
        "size_kb": round(filepath.stat().st_size / 1024, 1),
        "contract_types": set(),
        "symbols": set(),
        "indicators": set(),
        "risk_management": set(),
        "durations": set(),
        "barriers": set(),
        "has_virtual_loss": False,
        "has_soros": False,
        "has_martingale": False,
        "has_custom_barriers": False,
        "block_count": 0,
        "score": 0,
    }

    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        res["contract_types"] = []
        res["symbols"] = []
        res["indicators"] = []
        res["risk_management"] = []
        res["durations"] = []
        res["barriers"] = []
        return res

    # 1. Regex Fast Scanning
    ct_matches = re.findall(
        r"DIGITOVER|DIGITUNDER|DIGITEVEN|DIGITODD|DIGITMATCH|DIGITDIFF|\bCALL\b|\bPUT\b|HIGHER|LOWER|ONETOUCH|NOTOUCH",
        text,
        re.I,
    )
    for m in ct_matches:
        res["contract_types"].add(m.upper())

    sym_matches = re.findall(r"\b(R_\d+|1HZ\d+V|BOOM\d+|CRASH\d+|STEP\d+|JD\d+|RDBULL|RDBEAR|frx[A-Z]+)\b", text, re.I)
    for m in sym_matches:
        res["symbols"].add(m.upper())

    # Indicators
    ind_map = {
        "rsi": r"\brsi\b",
        "ema": r"\bema\b",
        "sma": r"\bsma\b",
        "macd": r"\bmacd\b",
        "bollinger": r"\b(bollinger|bband|bband_top|bband_bottom)\b",
        "stochastic": r"\b(stoch|stochastic)\b",
        "atr": r"\batr\b",
        "stddev": r"\bstddev\b",
        "tick_count": r"\b(tick_count|last_digit|digit_stat|frequency)\b",
    }
    for ind, pat in ind_map.items():
        if re.search(pat, text, re.I):
            res["indicators"].add(ind)

    # Risk Management
    if re.search(r"\b(soros|compound|lucro_composto)\b", text, re.I):
        res["has_soros"] = True
        res["risk_management"].add("soros")
    if re.search(r"\b(martingale|gale|multiplicador)\b", text, re.I):
        res["has_martingale"] = True
        res["risk_management"].add("martingale")
    if re.search(r"\b(virtual_loss|perda_virtual|loss_virtual|skip_loss)\b", text, re.I):
        res["has_virtual_loss"] = True
        res["risk_management"].add("virtual_loss")
    if re.search(r"\b(stop_loss|target_profit|meta_lucro|max_loss)\b", text, re.I):
        res["risk_management"].add("risk_limits")

    # Durations
    dur_matches = re.findall(r"""type=["']duration_unit["'][^>]*>([tms])""", text, re.I)
    for d in dur_matches:
        res["durations"].add(d.lower())

    if re.search(r"""type=["']barrier["']|offset|prediction""", text, re.I):
        res["has_custom_barriers"] = True

    # Fast block estimate
    res["block_count"] = len(re.findall(r"<block", text))

    # Calculate Score
    score = 0
    if res["block_count"] > 150:
        score += 30
    elif res["block_count"] > 50:
        score += 15
    elif res["block_count"] > 20:
        score += 5

    score += len(res["indicators"]) * 10
    if res["has_virtual_loss"]:
        score += 25
    if res["has_soros"]:
        score += 15
    if res["has_custom_barriers"]:
        score += 10
    if len(res["contract_types"]) > 2:
        score += 10

    res["score"] = score

    # Convert sets to lists
    res["contract_types"] = sorted(res["contract_types"])
    res["symbols"] = sorted(res["symbols"])
    res["indicators"] = sorted(res["indicators"])
    res["risk_management"] = sorted(res["risk_management"])
    res["durations"] = sorted(res["durations"])
    res["barriers"] = sorted(res["barriers"])

    return res


def run_batch_analysis() -> Dict[str, Any]:
    xml_files = sorted([str(p) for p in XML_DIR.glob("*.xml")] + [str(p) for p in XML_DIR.glob("*.XML")])
    xml_files = sorted(list(set(xml_files)))

    print(f"[+] Scanning {len(xml_files)} XML bots in parallel...")

    results: List[Dict[str, Any]] = []
    indicator_counts: Counter[str] = Counter()
    contract_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()

    with ProcessPoolExecutor() as executor:
        for info in executor.map(analyze_single_xml, xml_files, chunksize=50):
            results.append(info)
            for ind in info["indicators"]:
                indicator_counts[ind] += 1
            for ct in info["contract_types"]:
                contract_counts[ct] += 1
            for rk in info["risk_management"]:
                risk_counts[rk] += 1

    results.sort(key=lambda x: x["score"], reverse=True)

    summary = {
        "total_files_analyzed": len(xml_files),
        "most_popular_indicators": dict(indicator_counts.most_common()),
        "most_popular_contract_types": dict(contract_counts.most_common()),
        "risk_management_features": dict(risk_counts.most_common()),
        "top_ranked_bots": results[:30],
    }

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[+] Report saved to {OUTPUT_REPORT}")

    print("\n" + "=" * 90)
    print(f"TOP 15 STRATEGY CANDIDATES (Out of {len(xml_files)} analyzed)")
    print("=" * 90)
    print(f"{'#':<3} {'Filename':<45} {'Score':<6} {'Indicators':<20} {'Risk Features':<15}")
    print("-" * 90)

    for idx, bot in enumerate(results[:15], 1):
        fname = bot["filename"][:43] + ".." if len(bot["filename"]) > 43 else bot["filename"]
        inds = ", ".join(bot["indicators"][:3]) or "None"
        risk = ", ".join(bot["risk_management"]) or "Standard"
        print(f"{idx:<3} {fname:<45} {bot['score']:<6} {inds:<20} {risk:<15}")

    print("=" * 90)
    return summary


if __name__ == "__main__":
    run_batch_analysis()
