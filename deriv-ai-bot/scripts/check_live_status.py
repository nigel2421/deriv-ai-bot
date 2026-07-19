#!/usr/bin/env python3
"""Quick live bot health checklist."""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://deriv-ai-bot-842806243906.us-central1.run.app"


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return r.read().decode()


def main() -> int:
    health = get("/health").strip()
    j = json.loads(get("/status"))
    hb = j.get("heartbeat") or {}
    ds = j.get("deepseek") or {}
    an = j.get("analytics") or {}
    lf = an.get("last_filter") or j.get("last_filter") or {}
    learn = j.get("learning") or j.get("learner") or {}
    risk = j.get("risk") or {}
    lc = j.get("last_cycle_at")
    fresh = None
    if lc:
        try:
            t = datetime.fromisoformat(lc.replace("Z", "+00:00"))
            fresh = (datetime.now(timezone.utc) - t).total_seconds()
        except Exception as e:
            fresh = str(e)

    print("=== HEALTH ===", health)
    print("=== CORE ===")
    print("status:", j.get("status"))
    print("mode:", j.get("mode"))
    print("last_error:", j.get("last_error"))
    print("started_at:", j.get("started_at"))
    print("last_cycle_at:", lc, "age_sec:", round(fresh, 1) if isinstance(fresh, float) else fresh)
    print("symbols:", j.get("symbols"))
    print("buffers:", j.get("buffer_sizes"))
    print("closed_trades:", j.get("closed_trades"))
    print("=== RISK / HEARTBEAT ===")
    print("balance:", hb.get("balance"), hb.get("currency"))
    print("open_trades:", hb.get("open_trades"))
    print("daily_pnl:", hb.get("daily_pnl"))
    print("risk_paused:", hb.get("risk_paused"), hb.get("pause_reason"))
    print("min_confidence:", hb.get("min_confidence"))
    print("telegram_trading:", hb.get("telegram_trading"))
    print("session_stop_hit:", risk.get("session_stop_hit"))
    print("session_target_hit:", risk.get("session_target_hit"))
    print("=== DEEPSEEK ===")
    print(json.dumps(ds, indent=2)[:900] if ds else "missing")
    print("=== LEARNING ===")
    if learn:
        print(json.dumps(learn, indent=2)[:600])
    else:
        print("learning_keys=", hb.get("learning_keys"))
    print("=== LAST FILTER ===")
    if lf:
        print("allow:", lf.get("allow"), "rec:", lf.get("recommendation"), "action:", lf.get("action"))
        print("phase:", lf.get("learning_phase"), "regime:", lf.get("regime"))
        nt = lf.get("no_trade") or {}
        print("no_trade:", nt.get("status"), nt.get("reason"))
        print("ev:", lf.get("ev"), "dq:", lf.get("decision_quality"))
    else:
        print("none yet (ok if no candidate scored this cycle)")
    print("=== STRATEGIES ===")
    strats = j.get("strategies") or {}
    for s in j.get("symbols") or []:
        st = strats.get(s) or {}
        print(s, "tradeable=", st.get("tradeable"), "types=", st.get("allowed_types"))

    ok, bad = [], []

    def check(name, cond, detail=""):
        (ok if cond else bad).append(f"{name}: {detail if detail != '' else cond}")

    check("health_ok", health == "ok", health)
    check("status_running", j.get("status") == "running")
    check("no_last_error", not j.get("last_error"), repr(j.get("last_error")))
    check("cycle_fresh", isinstance(fresh, float) and fresh < 180, f"{fresh}s")
    check(
        "focused_symbols",
        j.get("symbols") == ["R_25", "R_50", "1HZ50V", "R_10"],
        str(j.get("symbols")),
    )
    bufs = j.get("buffer_sizes") or {}
    check(
        "buffers_filled",
        all(bufs.get(s, 0) >= 100 for s in (j.get("symbols") or [])),
        str(bufs),
    )
    check("min_conf_080", float(hb.get("min_confidence") or 0) == 0.8, str(hb.get("min_confidence")))
    check("deepseek_on", bool(ds.get("enabled") and ds.get("configured")), str(ds))
    check("deepseek_sk", str(ds.get("key_prefix") or "").startswith("sk-"), str(ds.get("key_prefix")))
    check("not_risk_paused", not hb.get("risk_paused"))
    check("balance_ok", float(hb.get("balance") or 0) > 10, str(hb.get("balance")))

    print("=== PASS ===")
    for x in ok:
        print(" ", x)
    print("=== FAIL/WARN ===")
    for x in bad:
        print(" ", x)
    print("OVERALL:", "ALL CHECKS PASSED" if not bad else "ISSUES FOUND")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
