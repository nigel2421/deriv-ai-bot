"""
Streamlit dashboard for Deriv AI Bot.

Reads local artifacts only (no live trading from the UI):
  - data/logs/bot.log
  - src/models/model_meta.json
  - data/training/backtest_summary.json / backtest_trades.csv
  - data/historical/*_ticks.csv

Run:
  streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "data" / "logs" / "bot.log"
META_PATH = ROOT / "src" / "models" / "model_meta.json"
SCHEMA_PATH = ROOT / "src" / "models" / "feature_schema.json"
BT_SUMMARY = ROOT / "data" / "training" / "backtest_summary.json"
BT_TRADES = ROOT / "data" / "training" / "backtest_trades.csv"
HIST_DIR = ROOT / "data" / "historical"


st.set_page_config(
    page_title="Deriv AI Bot",
    page_icon="📈",
    layout="wide",
)
st.title("📈 Deriv AI Trading Bot — Local Dashboard")
st.caption(f"Project root: `{ROOT}` · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tail_log(path: Path, n: int = 80) -> str:
    if not path.is_file():
        return "(no log file yet — run the bot first)"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as e:
        return f"(failed to read log: {e})"


# --- Model metrics ---
st.header("Model")
meta = _load_json(META_PATH)
schema = _load_json(SCHEMA_PATH)

c1, c2, c3, c4 = st.columns(4)
acc = meta.get("ensemble_test_accuracy") or meta.get("xgb_test_accuracy")
c1.metric("Holdout accuracy", f"{acc:.1%}" if isinstance(acc, (int, float)) else "—")
c2.metric("Baseline", f"{meta.get('baseline_accuracy', 0):.1%}" if meta.get("baseline_accuracy") is not None else "—")
c3.metric("Features", meta.get("feature_count") or schema.get("n_features") or "—")
c4.metric("Gate", "PASS" if meta.get("passed_gate") else ("FAIL" if meta else "—"))

with st.expander("model_meta.json"):
    st.json(meta or {"info": "No model_meta.json — run scripts/train_model.py"})
with st.expander("feature_schema.json"):
    st.json(schema or {"info": "No feature schema yet"})


# --- Backtest ---
st.header("Backtest")
bt = _load_json(BT_SUMMARY)
if bt:
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Trades", bt.get("trades", "—"))
    wr = bt.get("win_rate")
    b2.metric("Win rate", f"{wr:.1%}" if isinstance(wr, (int, float)) else "—")
    b3.metric("PnL", f"{bt.get('total_profit', 0):.2f}")
    b4.metric("Max DD", f"{bt.get('max_drawdown', 0):.2f}")

    eq = bt.get("equity_curve") or []
    if eq:
        st.subheader("Equity curve")
        st.line_chart(pd.DataFrame({"equity": eq}))
else:
    st.info("No backtest summary. Run: `python scripts/backtest.py --export ...`")

if BT_TRADES.is_file():
    trades = pd.read_csv(BT_TRADES)
    st.subheader("Recent backtest trades")
    st.dataframe(trades.tail(50), use_container_width=True)
    if "profit" in trades.columns:
        st.subheader("Trade PnL histogram")
        st.bar_chart(trades["profit"].value_counts().sort_index())


# --- Market data ---
st.header("Historical ticks")
tick_files = sorted(HIST_DIR.glob("*_ticks.csv")) if HIST_DIR.is_dir() else []
if tick_files:
    choice = st.selectbox("Symbol file", [p.name for p in tick_files])
    path = HIST_DIR / choice
    tdf = pd.read_csv(path)
    st.write(f"{len(tdf)} rows from `{path.name}`")
    if "quote" in tdf.columns:
        st.line_chart(tdf["quote"].tail(300))
    st.dataframe(tdf.tail(20), use_container_width=True)
else:
    st.info("No tick CSVs. Run: `python scripts/data_collector.py`")


# --- Logs ---
st.header("Bot log (tail)")
n = st.slider("Lines", 20, 300, 80)
st.code(_tail_log(LOG_PATH, n), language="log")

st.divider()
st.markdown(
    "**Tips:** demo mode only until validated · "
    "`python src/main.py --mode demo` · "
    "`docker compose up --build`"
)
