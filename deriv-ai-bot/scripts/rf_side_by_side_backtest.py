#!/usr/bin/env python3
"""
Side-by-side Rise/Fall backtest:

  A) OLD digit-weight profile for CALL/PUT
     (digit_entropy heavy — previous architecture)

  B) NEW directional weights
     (momentum / trend / vol / persistence / dir-entropy)

Runs synthetic + optional historical tick replay.
Reports win rate, profit factor, max drawdown for both.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.contract_profiles import contract_clarity, build_metric_vector
from src.analytics.rise_fall_engine import analyze_rise_fall, rf_pattern_clarity
from src.analytics.momentum_persistence_engine import analyze_momentum_persistence
from src.strategy.chart_tools import quotes_from_ticks

# --- Weight profiles ---
OLD_DIGIT_WEIGHTS = {
    "up_down_entropy": 0.35,
    "momentum": 0.30,
    "streak_entropy": 0.15,
    "digit_entropy": 0.10,
    "stability": 0.10,
}

NEW_DIRECTIONAL_WEIGHTS = {
    "momentum": 0.35,
    "trend_strength": 0.25,
    "volatility_score": 0.20,
    "persistence": 0.10,
    "directional_entropy": 0.10,
}


def synth_ticks(
    n: int = 2000,
    *,
    seed: int = 42,
    regime: str = "mixed",
) -> List[Dict[str, Any]]:
    """Generate synthetic quotes with controllable directional regimes."""
    rng = random.Random(seed)
    q = 100.0
    ticks = []
    for i in range(n):
        if regime == "bull":
            drift = 0.012
        elif regime == "bear":
            drift = -0.012
        else:
            # Switch every ~200 ticks
            block = (i // 200) % 4
            drift = {0: 0.01, 1: -0.008, 2: 0.003, 3: -0.012}.get(block, 0.0)
        noise = rng.uniform(-0.015, 0.015)
        q = max(1.0, q + drift + noise)
        # last-digit variety for digit-weight path
        digit = rng.randint(0, 9)
        quote = float(f"{int(q)}.{digit}")
        # preserve trend via integer part mostly
        quote = q
        ticks.append({"epoch": 1_700_000_000 + i, "quote": round(quote, 4), "symbol": "R_50"})
    return ticks


def load_historical(path: Path, symbol: str = "R_50", limit: int = 3000) -> List[Dict[str, Any]]:
    import pandas as pd

    df = pd.read_csv(path)
    if "quote" not in df.columns:
        # try common alternatives
        for c in ("price", "close", "bid"):
            if c in df.columns:
                df = df.rename(columns={c: "quote"})
                break
    rows = []
    for i, r in df.tail(limit).iterrows():
        rows.append(
            {
                "epoch": int(r.get("epoch") or i),
                "quote": float(r["quote"]),
                "symbol": symbol,
            }
        )
    return rows


def score_old(ticks: Sequence[Dict[str, Any]], contract: str) -> Dict[str, Any]:
    """Old digit-weight clarity for CALL/PUT (manual weighted sum)."""
    from src.analytics.rolling_entropy import feed_ticks

    roll = feed_ticks("_bt", list(ticks)[-200:])
    metrics = build_metric_vector(
        rolling=roll,
        momentum=float(roll.get("momentum_score") or 50),
        stability=float(roll.get("stability_score") or 55),
    )
    # Manual OLD profile: Σ metric × weight
    s = 0.0
    wsum = 0.0
    for k, w in OLD_DIGIT_WEIGHTS.items():
        val = float(metrics.get(k) or metrics.get(k.replace("_entropy", "")) or 50)
        # map aliases
        if k == "up_down_entropy":
            val = float(metrics.get("up_down_entropy") or metrics.get("directional_entropy") or 50)
        s += val * w
        wsum += w
    score = s / (wsum or 1.0)
    return {
        "score": float(score),
        "metrics": metrics,
        "mode": "old_digit_weights",
    }


def score_new(ticks: Sequence[Dict[str, Any]], contract: str) -> Dict[str, Any]:
    """New directional RF engine + MP."""
    rf = analyze_rise_fall(ticks, contract_type=contract, hpp=55.0)
    mp = analyze_momentum_persistence(
        ticks, symbol="R_50", contract_type=contract, note_velocity=False
    )
    # Blend RF score with MP
    rf_s = float(rf.get("rf_score") or 50)
    mp_s = float(mp.get("mp_score") or 50)
    clarity = rf_pattern_clarity(rf)
    score = 0.55 * rf_s + 0.30 * mp_s + 0.15 * clarity
    return {
        "score": score,
        "rf": rf,
        "mp": mp,
        "mode": "new_directional",
    }


def settle_rf(
    ticks: Sequence[Dict[str, Any]],
    i: int,
    contract: str,
    horizon: int = 5,
) -> Optional[bool]:
    """Win if price moved in predicted direction after horizon ticks."""
    if i + horizon >= len(ticks):
        return None
    a = float(ticks[i]["quote"])
    b = float(ticks[i + horizon]["quote"])
    if abs(b - a) < 1e-12:
        return None  # push
    up = b > a
    if contract in {"CALL", "RISE", "HIGHER"}:
        return up
    if contract in {"PUT", "FALL", "LOWER"}:
        return not up
    return None


def run_mode(
    ticks: List[Dict[str, Any]],
    *,
    mode: str,
    min_score: float = 70.0,
    min_confidence: float = 0.80,
    horizon: int = 5,
    step: int = 3,
    warmup: int = 80,
    stake: float = 1.0,
    payout: float = 0.95,
) -> Dict[str, Any]:
    balance = 1000.0
    peak = balance
    max_dd = 0.0
    wins = losses = 0
    gross_p = gross_l = 0.0
    equity = [balance]
    trades = []

    for i in range(warmup, len(ticks) - horizon, step):
        window = ticks[max(0, i - 120) : i + 1]
        # Pick side from short momentum
        quotes = quotes_from_ticks(window, n=30)
        if len(quotes) < 10:
            continue
        ups = sum(1 for j in range(1, len(quotes)) if quotes[j] > quotes[j - 1])
        downs = sum(1 for j in range(1, len(quotes)) if quotes[j] < quotes[j - 1])
        if ups == downs:
            continue
        contract = "CALL" if ups > downs else "PUT"
        conf = abs(ups - downs) / max(1, ups + downs)  # 0..1
        # Map to 0.5–1.0 confidence band
        conf = 0.50 + 0.50 * conf
        if conf < min_confidence:
            continue

        if mode == "old":
            sc = score_old(window, contract)
        else:
            sc = score_new(window, contract)

        if sc["score"] < min_score:
            continue

        # Directional gate: new mode needs vol tradeable when available
        if mode == "new":
            rf = sc.get("rf") or {}
            if rf.get("vol_tradeable") is False:
                continue
            # oriented: score already contract-specific

        outcome = settle_rf(ticks, i, contract, horizon=horizon)
        if outcome is None:
            continue

        if outcome:
            profit = stake * payout
            wins += 1
            gross_p += profit
        else:
            profit = -stake
            losses += 1
            gross_l += stake
        balance += profit
        peak = max(peak, balance)
        dd = peak - balance
        max_dd = max(max_dd, dd)
        equity.append(balance)
        trades.append(
            {
                "i": i,
                "contract": contract,
                "score": round(sc["score"], 1),
                "conf": round(conf, 3),
                "win": outcome,
                "profit": profit,
            }
        )

    n = wins + losses
    wr = wins / n if n else 0.0
    pf = (gross_p / gross_l) if gross_l > 1e-9 else (10.0 if gross_p > 0 else 0.0)
    return {
        "mode": mode,
        "trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr * 100, 2),
        "profit": round(balance - 1000.0, 2),
        "final_balance": round(balance, 2),
        "profit_factor": round(pf, 3),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd / peak * 100, 2) if peak else 0.0,
        "min_confidence": min_confidence,
        "min_score": min_score,
        "sample_trades": trades[:5],
    }


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pf_improved": b["profit_factor"] >= a["profit_factor"],
        "wr_improved_or_stable": b["win_rate"] >= a["win_rate"] - 1.0,
        "dd_improved_or_stable": b["max_drawdown"] <= a["max_drawdown"] + 0.5,
        "delta_pf": round(b["profit_factor"] - a["profit_factor"], 3),
        "delta_wr": round(b["win_rate"] - a["win_rate"], 2),
        "delta_dd": round(b["max_drawdown"] - a["max_drawdown"], 2),
        "delta_pnl": round(b["profit"] - a["profit"], 2),
        "winner": "new_directional"
        if (
            b["profit_factor"] > a["profit_factor"]
            or (
                abs(b["profit_factor"] - a["profit_factor"]) < 0.05
                and b["profit"] > a["profit"]
            )
        )
        else "old_digit_weights",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="RF side-by-side backtest")
    ap.add_argument("--ticks", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-confidence", type=float, default=0.80)
    ap.add_argument("--min-score", type=float, default=68.0)
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--json-out", type=str, default="data/rf_side_by_side_result.json")
    args = ap.parse_args()

    if args.csv and Path(args.csv).is_file():
        ticks = load_historical(Path(args.csv))
        source = f"csv:{args.csv}"
    else:
        ticks = synth_ticks(args.ticks, seed=args.seed, regime="mixed")
        source = f"synthetic n={args.ticks} seed={args.seed}"

    print("=" * 60)
    print("RF SIDE-BY-SIDE BACKTEST")
    print(f"Source: {source}")
    print(f"min_confidence={args.min_confidence} (unified)")
    print("=" * 60)

    old = run_mode(
        ticks,
        mode="old",
        min_score=args.min_score,
        min_confidence=args.min_confidence,
    )
    new = run_mode(
        ticks,
        mode="new",
        min_score=args.min_score,
        min_confidence=args.min_confidence,
    )
    cmp = compare(old, new)

    print("\n--- OLD (digit weights on CALL/PUT) ---")
    print(json.dumps({k: old[k] for k in old if k != "sample_trades"}, indent=2))
    print("\n--- NEW (directional / momentum / persistence) ---")
    print(json.dumps({k: new[k] for k in new if k != "sample_trades"}, indent=2))
    print("\n--- COMPARISON ---")
    print(json.dumps(cmp, indent=2))

    out = {
        "source": source,
        "old": old,
        "new": new,
        "comparison": cmp,
        "pass_criteria": {
            "profit_factor_improved_or_equal": cmp["pf_improved"],
            "wr_stable_or_up": cmp["wr_improved_or_stable"],
            "dd_stable_or_down": cmp["dd_improved_or_stable"],
        },
    }
    path = Path(args.json_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")
    print(f"Winner: {cmp['winner']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
