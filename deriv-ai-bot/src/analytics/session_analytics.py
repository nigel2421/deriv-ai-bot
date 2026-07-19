"""
Best/worst hours and markets from recorded trades.
Feeds skip recommendations for weak hours/symbols.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_PATH = Path("data/session_analytics.json")


def _hour_from_ts(ts: Any) -> Optional[int]:
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        else:
            s = str(ts).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.hour
    except Exception:
        return None


class SessionAnalytics:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.trades: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.trades = list(data.get("trades") or [])[-5000:]
        except Exception:
            self.trades = []

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {"trades": self.trades[-5000:], "updated_at": time.time()},
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def record(
        self,
        *,
        symbol: str,
        contract_type: str,
        is_win: bool,
        profit: float,
        ts: Optional[Any] = None,
        stake: Optional[float] = None,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.trades.append(
            {
                "symbol": symbol,
                "contract_type": contract_type,
                "is_win": bool(is_win),
                "profit": float(profit),
                "stake": stake,
                "ts": ts or datetime.now(timezone.utc).isoformat(),
                "indicators": indicators or {},
            }
        )
        if len(self.trades) > 5000:
            self.trades = self.trades[-5000:]
        self.save()

    def _wr(self, rows: Sequence[Dict[str, Any]]) -> float:
        if not rows:
            return 0.5
        w = sum(1 for r in rows if r.get("is_win") or float(r.get("profit") or 0) > 0)
        return w / len(rows)

    def hour_stats(self) -> List[Dict[str, Any]]:
        by_h: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for t in self.trades:
            h = _hour_from_ts(t.get("ts"))
            if h is not None:
                by_h[h].append(t)
        out = []
        for h in range(24):
            rows = by_h.get(h) or []
            if not rows:
                continue
            out.append(
                {
                    "hour_utc": h,
                    "trades": len(rows),
                    "win_rate": round(self._wr(rows), 3),
                    "pnl": round(sum(float(r.get("profit") or 0) for r in rows), 2),
                }
            )
        out.sort(key=lambda x: x["win_rate"], reverse=True)
        return out

    def symbol_stats(self) -> List[Dict[str, Any]]:
        by_s: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in self.trades:
            by_s[str(t.get("symbol") or "?")].append(t)
        out = []
        for sym, rows in by_s.items():
            out.append(
                {
                    "symbol": sym,
                    "trades": len(rows),
                    "win_rate": round(self._wr(rows), 3),
                    "pnl": round(sum(float(r.get("profit") or 0) for r in rows), 2),
                }
            )
        out.sort(key=lambda x: x["win_rate"], reverse=True)
        return out

    def skip_hour(self, hour_utc: Optional[int] = None, min_trades: int = 15) -> tuple:
        if hour_utc is None:
            hour_utc = datetime.now(timezone.utc).hour
        for row in self.hour_stats():
            if row["hour_utc"] == hour_utc and row["trades"] >= min_trades:
                if row["win_rate"] < 0.42:
                    return True, f"worst_hour_utc_{hour_utc}_wr={row['win_rate']:.0%}"
        return False, ""

    def skip_symbol(self, symbol: str, min_trades: int = 12) -> tuple:
        for row in self.symbol_stats():
            if row["symbol"] == symbol and row["trades"] >= min_trades:
                if row["win_rate"] < 0.40 or row["pnl"] < 0 and row["win_rate"] < 0.48:
                    return True, f"weak_symbol_{symbol}_wr={row['win_rate']:.0%}"
        return False, ""

    def insights(self) -> Dict[str, Any]:
        hours = self.hour_stats()
        syms = self.symbol_stats()
        best_h = hours[0] if hours else None
        worst_h = hours[-1] if hours else None
        best_s = syms[0] if syms else None
        worst_s = syms[-1] if syms else None
        lines = []
        if best_h and best_h["trades"] >= 10:
            lines.append(
                f"Best hour (UTC): {best_h['hour_utc']:02d}:00 WR={best_h['win_rate']:.0%} (n={best_h['trades']})"
            )
        if worst_h and worst_h["trades"] >= 10:
            lines.append(
                f"Worst hour (UTC): {worst_h['hour_utc']:02d}:00 WR={worst_h['win_rate']:.0%}"
            )
        if best_s and best_s["trades"] >= 8:
            lines.append(
                f"Best index: {best_s['symbol']} WR={best_s['win_rate']:.0%} PnL={best_s['pnl']}"
            )
        if worst_s and worst_s["trades"] >= 8:
            lines.append(
                f"Worst index: {worst_s['symbol']} WR={worst_s['win_rate']:.0%} PnL={worst_s['pnl']}"
            )
        if len(self.trades) >= 100:
            lines.append(f"Session recorder: {len(self.trades)} trades stored.")
        return {
            "n_trades": len(self.trades),
            "best_hour": best_h,
            "worst_hour": worst_h,
            "best_symbol": best_s,
            "worst_symbol": worst_s,
            "hours": hours[:5],
            "symbols": syms[:8],
            "lines": lines,
        }

    def snapshot(self) -> Dict[str, Any]:
        return self.insights()
