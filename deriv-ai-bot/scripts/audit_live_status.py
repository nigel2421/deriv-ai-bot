"""Pull Cloud Run /status and summarize learning + trade gaps."""
from __future__ import annotations

import json
import urllib.request
from collections import Counter

URL = "https://deriv-ai-bot-842806243906.us-central1.run.app/status"


def main() -> None:
    with urllib.request.urlopen(URL, timeout=90) as r:
        s = json.load(r)

    h = s.get("heartbeat") or {}
    learn = s.get("learning") or {}
    recent = s.get("recent_trades") or []
    risk = s.get("risk") or {}

    print("=== SESSION ===")
    print(
        "balance",
        h.get("balance"),
        "daily_pnl",
        h.get("daily_pnl"),
        "closed",
        s.get("closed_trades"),
        "session_start",
        risk.get("session_start_balance"),
    )
    print(
        "min_conf",
        s.get("min_confidence"),
        "minute",
        s.get("enable_minute"),
        s.get("minute_duration"),
        "stake_mode",
        s.get("stake_mode"),
    )
    print("symbols", s.get("symbols"))
    print(
        "learning total",
        learn.get("total_recorded"),
        "W/L",
        learn.get("global_wins"),
        learn.get("global_losses"),
        "phase",
        learn.get("phase"),
        "keys",
        learn.get("keys"),
    )
    print("preferred", learn.get("preferred"))
    print("top setups:")
    for t in (learn.get("top") or [])[:20]:
        print(" ", t)

    closed = [t for t in recent if t.get("status") in ("win", "loss", "push")]
    fails = [t for t in recent if "fail" in str(t.get("status", ""))]
    print("=== RECENT WINDOW ===")
    print("closed", len(closed), "fails", len(fails))
    print(
        "by family/horizon/dur",
        dict(
            Counter(
                (
                    t.get("family"),
                    t.get("horizon"),
                    t.get("duration"),
                    t.get("duration_unit"),
                )
                for t in closed
            )
        ),
    )
    print("by symbol", dict(Counter(t.get("symbol") for t in closed)))
    print("by type", dict(Counter(t.get("contract_type") for t in closed)))
    pnl = sum(float(t.get("profit") or 0) for t in closed)
    wins = sum(1 for t in closed if t.get("status") == "win")
    print("recent closed pnl", round(pnl, 2), "wr", f"{wins}/{len(closed)}")
    confs = [float(t.get("confidence") or 0) for t in closed if t.get("confidence")]
    if confs:
        print(
            "avg conf",
            round(sum(confs) / len(confs), 3),
            "min",
            round(min(confs), 3),
            "max",
            round(max(confs), 3),
        )
    print(
        "fail reasons",
        dict(
            Counter(
                (
                    t.get("symbol"),
                    t.get("offer_reason") or str(t.get("error", ""))[:50],
                    t.get("duration"),
                    t.get("duration_unit"),
                    t.get("family"),
                )
                for t in fails
            )
        ),
    )

    fx = [x for x in (s.get("symbols") or []) if str(x).startswith("frx")]
    print("FX symbols", fx)
    fx_trades = [t for t in recent if str(t.get("symbol", "")).startswith("frx")]
    print("FX recent trades count", len(fx_trades), fx_trades[:5])

    strats = s.get("strategies") or {}
    for k in fx:
        st = strats.get(k) or {}
        print(
            "FX strategy",
            k,
            "allowed",
            st.get("allowed_types"),
            "tradeable",
            st.get("tradeable"),
        )

    ds = s.get("deepseek") or {}
    rec = ds.get("recommendation") or {}
    print("=== DEEPSEEK ===")
    print("closes_since", ds.get("closes_since_analysis"), "due", ds.get("due_setups"))
    print("summary", rec.get("summary"))
    print("hints", rec.get("learning_hints"))
    print("type_multipliers", ds.get("type_multipliers"))
    print("bans", ds.get("bans"))

    cal = s.get("calibration") or {}
    print("=== CALIBRATION ===")
    if isinstance(cal, dict):
        print(
            "cum",
            cal.get("cumulative_trades"),
            "auto",
            cal.get("auto_deflation_enabled"),
            "overall_err",
            cal.get("overall_error"),
        )
        for row in cal.get("rows") or []:
            print(" ", row)

    aud = s.get("ai_auditor")
    print("=== AUDITOR ===", (aud or {}).get("type") if isinstance(aud, dict) else aud)
    if isinstance(aud, dict):
        print("wr", aud.get("overall_win_rate"), "n", aud.get("trades_analyzed"))
        print("recs", (aud.get("recommendations") or [])[:5])


if __name__ == "__main__":
    main()
