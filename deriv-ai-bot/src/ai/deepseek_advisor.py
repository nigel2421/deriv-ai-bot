"""
DeepSeek advisor — analyzes trade runs and recommends strategy improvements.

Uses the OpenAI-compatible DeepSeek chat API (httpx, no openai package required).
Recommendations feed the adaptive learner and operator dashboard.

Sample discipline (token efficiency):
- Do not call the API on tiny samples (default min 20 closed trades globally,
  or 12+ on a single market|strategy bucket).
- Auto cadence: every N global closes (default 20), or when a setup bucket
  reaches per-setup min closes since last analysis of that bucket.
- Payload is always structured per market / per strategy (family) with
  contract-type breakdowns so verdicts match what the bot actually trades.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import httpx

logger = logging.getLogger(__name__)

DEFAULT_SKILL_PATH = Path(__file__).resolve().parents[2] / "skills" / "deepseek-trading" / "SKILL.md"
DEFAULT_CACHE_PATH = Path("data/deepseek_recommendations.json")

# Defaults tuned for signal quality vs token cost
DEFAULT_ANALYZE_EVERY = 20          # global closes between full runs
DEFAULT_MIN_SAMPLE = 20             # min closed trades to spend tokens
DEFAULT_MIN_PER_SETUP = 12          # min per market|strategy bucket

# Fallback system prompt if SKILL.md is missing
_FALLBACK_SYSTEM = """You are the DeepSeek trading advisor for a Deriv multi-market bot
(synthetic vols, Boom/Crash, Jump, daily reset, etc.). Protect capital first.

You receive data GROUPED per market and per strategy family (digits, rise_fall,
minute_rise_fall). Only recommend for buckets that appear in the payload with
enough samples. Never invent symbols or contract types the bot did not trade.

Rules:
- Risk 1–2% of balance per trade max.
- Session stop-loss: 5–10% of session-start balance (operator-set).
- Session profit target: 1:3 risk:reward (target = stop_loss_amount × 3).
- Prefer trend-following for CALL/PUT; digits scored separately.
- Only high-quality setups; ban losing symbol|type pairs with enough samples.

Respond with JSON only:
{
  "summary": "1-3 sentences",
  "risk_score": 0-100,
  "trade_type_analysis": [
    {
      "symbol": "R_25",
      "family": "rise_fall",
      "contract_type": "CALL",
      "verdict": "keep|reduce|ban",
      "reason": "why (cite n, wr, pnl from that bucket)",
      "suggested_confidence_mult": 0.5-1.25
    }
  ],
  "strategy_changes": ["..."],
  "stake_advice": {"action": "keep|lower|raise", "pct_of_balance": 1.0, "reason": "..."},
  "session_advice": {"stop_loss_pct": 5.0, "target_rr": 3.0, "reason": "..."},
  "learning_hints": ["..."]
}
Every trade_type_analysis row MUST include symbol + contract_type (and family when known).
Do not emit type-only rows without a symbol.
"""


def load_skill_prompt(path: Optional[Path] = None) -> str:
    p = Path(path) if path else DEFAULT_SKILL_PATH
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Could not read DeepSeek skill file %s: %s", p, e)
    return _FALLBACK_SYSTEM


def _setup_key(symbol: str, family: str, contract_type: str = "") -> str:
    sym = str(symbol or "").strip() or "?"
    fam = str(family or "unknown").strip().lower() or "unknown"
    ct = str(contract_type or "").strip().upper()
    if ct:
        return f"{sym}|{fam}|{ct}"
    return f"{sym}|{fam}"


def _trade_family(t: Dict[str, Any]) -> str:
    fam = str(t.get("family") or "").strip().lower()
    if fam:
        return fam
    ct = str(t.get("contract_type") or "").upper()
    if ct.startswith("DIGIT"):
        return "digits"
    if ct in {"CALL", "PUT"}:
        return "rise_fall"
    return "unknown"


class DeepSeekAdvisor:
    """
    Calls DeepSeek to review closed trades + learning stats and produce
    structured recommendations for risk and trade-type weighting.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        enabled: bool = True,
        timeout_sec: float = 45.0,
        skill_path: Optional[Path] = None,
        cache_path: Optional[Path] = None,
        analyze_every: int = DEFAULT_ANALYZE_EVERY,
        min_sample: int = DEFAULT_MIN_SAMPLE,
        min_per_setup: int = DEFAULT_MIN_PER_SETUP,
    ):
        raw_key = (api_key or "").strip() or None
        self.api_key = raw_key
        self.base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self.model = model or "deepseek-v4-flash"
        self.timeout_sec = float(timeout_sec)
        self.skill_path = Path(skill_path) if skill_path else DEFAULT_SKILL_PATH
        self.cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
        self.analyze_every = max(0, int(analyze_every))
        self.min_sample = max(5, int(min_sample))
        self.min_per_setup = max(5, int(min_per_setup))
        self.system_prompt = load_skill_prompt(self.skill_path)
        self.last_recommendation: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self.closes_since_analysis = 0
        # Closes since last analysis of that market|family (or market|family|type)
        self._setup_closes: Dict[str, int] = {}
        self._type_multipliers: Dict[str, float] = {}
        self._bans: set[str] = set()
        self._preferred: set[str] = set()
        self.key_valid = False
        if raw_key:
            if raw_key.startswith("sk-"):
                self.key_valid = True
            elif raw_key.startswith("AIza"):
                self.last_error = (
                    "DEEPSEEK_API_KEY looks like a Google API key (AIza…). "
                    "Use a DeepSeek sk-… key from platform.deepseek.com — "
                    "update Secret Manager deepseek-api-key before deploy."
                )
                logger.error(self.last_error)
            else:
                self.key_valid = True
                logger.warning(
                    "DEEPSEEK_API_KEY does not start with sk- (prefix=%s…) — "
                    "confirm it is a real DeepSeek key",
                    raw_key[:6],
                )
        self.enabled = bool(enabled) and bool(self.api_key) and self.key_valid
        if enabled and raw_key and not self.key_valid:
            self.enabled = False
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.is_file():
            return
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self.last_recommendation = data.get("recommendation")
            mults = data.get("type_multipliers") or {}
            self._type_multipliers = {
                str(k).upper(): float(v)
                for k, v in mults.items()
                if isinstance(v, (int, float))
            }
            self._bans = {str(k).upper() for k in (data.get("bans") or [])}
            self._preferred = {
                str(k).upper() for k in (data.get("preferred") or [])
            }
            sc = data.get("setup_closes") or {}
            self._setup_closes = {
                str(k): int(v) for k, v in sc.items() if int(v) > 0
            }
            if not self._bans and self._type_multipliers:
                self._rebuild_ban_pref_from_mults()
            logger.info(
                "DeepSeekAdvisor loaded cache (%d type mults, %d bans) from %s",
                len(self._type_multipliers),
                len(self._bans),
                self.cache_path,
            )
        except Exception as e:
            logger.debug("DeepSeek cache load failed: %s", e)

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "recommendation": self.last_recommendation,
                "type_multipliers": self._type_multipliers,
                "bans": sorted(self._bans),
                "preferred": sorted(self._preferred),
                "setup_closes": dict(self._setup_closes),
                "closes_since_analysis": self.closes_since_analysis,
                "updated_at": time.time(),
            }
            self.cache_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.debug("DeepSeek cache save failed: %s", e)

    def is_ready(self) -> bool:
        return bool(self.enabled and self.api_key)

    def note_closed_trade(
        self,
        symbol: Optional[str] = None,
        family: Optional[str] = None,
        contract_type: Optional[str] = None,
    ) -> bool:
        """
        Increment counters. Returns True when an analysis should run.

        Triggers (any):
        - Global closes_since_analysis >= analyze_every (default 20)
        - A market|strategy bucket has >= min_per_setup closes since last
          analysis of that bucket (default 12)
        """
        if not self.is_ready() or self.analyze_every <= 0:
            return False
        self.closes_since_analysis += 1
        fam = str(family or "unknown").strip().lower() or "unknown"
        sym = str(symbol or "").strip()
        ct = str(contract_type or "").strip().upper()
        if sym:
            mk = _setup_key(sym, fam)
            self._setup_closes[mk] = int(self._setup_closes.get(mk) or 0) + 1
            if ct:
                tk = _setup_key(sym, fam, ct)
                self._setup_closes[tk] = int(self._setup_closes.get(tk) or 0) + 1

        due_setup = any(
            n >= self.min_per_setup for n in self._setup_closes.values()
        )
        due_global = self.closes_since_analysis >= self.analyze_every
        return bool(due_setup or due_global)

    def due_setup_keys(self) -> List[str]:
        """Market|family keys that hit the per-setup threshold."""
        out = []
        for k, n in self._setup_closes.items():
            if n < self.min_per_setup:
                continue
            # Prefer market|family (2 parts) over type keys for scoping
            parts = k.split("|")
            if len(parts) == 2:
                out.append(k)
            elif len(parts) == 3:
                mk = f"{parts[0]}|{parts[1]}"
                if mk not in out:
                    out.append(mk)
        return out

    def mark_analyzed(self, setup_keys: Optional[Sequence[str]] = None) -> None:
        """Reset global and (optionally) per-setup counters after a successful run."""
        self.closes_since_analysis = 0
        if setup_keys is None:
            self._setup_closes.clear()
        else:
            keys = set(setup_keys)
            # Also clear type-level children of market|family
            drop = []
            for k in self._setup_closes:
                if k in keys:
                    drop.append(k)
                    continue
                parts = k.split("|")
                if len(parts) == 3 and f"{parts[0]}|{parts[1]}" in keys:
                    drop.append(k)
            for k in drop:
                self._setup_closes.pop(k, None)
        self._save_cache()

    def confidence_multiplier(self, symbol: str, contract_type: str) -> float:
        """Per trade-type multiplier from last DeepSeek analysis (default 1.0)."""
        if not self._type_multipliers:
            return 1.0
        key = f"{symbol}|{str(contract_type).upper()}"
        if key in self._type_multipliers:
            return max(0.5, min(1.25, self._type_multipliers[key]))
        ct = str(contract_type).upper()
        if ct in self._type_multipliers:
            return max(0.5, min(1.25, self._type_multipliers[ct]))
        return 1.0

    def is_banned(self, symbol: str, contract_type: str) -> bool:
        """Hard skip when DeepSeek banned this symbol|type (or type globally)."""
        ct = str(contract_type or "").upper()
        sym = str(symbol or "").strip()
        if not ct:
            return False
        keys = [ct]
        if sym:
            keys.insert(0, f"{sym}|{ct}")
        for k in keys:
            if k in self._bans:
                return True
            if k in self._type_multipliers and self._type_multipliers[k] <= 0.55:
                return True
        return False

    def selection_boost(self, symbol: str, contract_type: str) -> float:
        """Score delta for trade_selector."""
        ct = str(contract_type or "").upper()
        sym = str(symbol or "").strip()
        key = f"{sym}|{ct}" if sym else ct
        if self.is_banned(symbol, contract_type):
            return -0.5
        mult = self.confidence_multiplier(symbol, contract_type)
        if key in self._preferred or ct in self._preferred:
            return 0.04 + max(0.0, (mult - 1.0) * 0.08)
        if mult >= 1.08:
            return 0.03
        if mult <= 0.85:
            return -0.04
        if mult < 0.95:
            return -0.02
        return 0.0

    def preferred_symbols(self) -> List[str]:
        """Symbols DeepSeek prefers (for scan ordering)."""
        out: List[str] = []
        seen = set()
        for k in list(self._preferred) + list(self._type_multipliers.keys()):
            if "|" not in k:
                continue
            sym = k.split("|", 1)[0].strip()
            mult = float(self._type_multipliers.get(k) or 1.0)
            if not sym or mult < 1.0:
                continue
            if sym not in seen:
                seen.add(sym)
                out.append(sym)
        return out

    def _rebuild_ban_pref_from_mults(self) -> None:
        self._bans = {
            k for k, v in self._type_multipliers.items() if float(v) <= 0.55
        }
        self._preferred = {
            k for k, v in self._type_multipliers.items() if float(v) >= 1.05
        }

    @staticmethod
    def build_market_strategy_buckets(
        trades: Sequence[Dict[str, Any]],
        *,
        min_n: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Group closed trades by market (symbol) + strategy family, with
        contract_type breakdowns. This is what DeepSeek should analyze.
        """
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in trades:
            if not isinstance(t, dict):
                continue
            status = str(t.get("status") or "").lower()
            if status not in {"win", "loss", "push"}:
                # allow profit-based inference
                if t.get("profit") is None:
                    continue
            sym = str(t.get("symbol") or "").strip() or "?"
            fam = _trade_family(t)
            groups[f"{sym}|{fam}"].append(t)

        buckets: List[Dict[str, Any]] = []
        for key, rows in groups.items():
            sym, fam = key.split("|", 1)
            wins = losses = pushes = 0
            pnl = 0.0
            by_type: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                st = str(r.get("status") or "").lower()
                try:
                    profit = float(r.get("profit") or 0)
                except (TypeError, ValueError):
                    profit = 0.0
                if st not in {"win", "loss", "push"}:
                    st = "win" if profit > 0 else ("push" if profit == 0 else "loss")
                if st == "win":
                    wins += 1
                elif st == "loss":
                    losses += 1
                else:
                    pushes += 1
                pnl += profit
                ct = str(r.get("contract_type") or "?").upper()
                bt = by_type.setdefault(
                    ct,
                    {"contract_type": ct, "n": 0, "wins": 0, "losses": 0, "pnl": 0.0},
                )
                bt["n"] += 1
                bt["pnl"] = float(bt["pnl"]) + profit
                if st == "win":
                    bt["wins"] += 1
                elif st == "loss":
                    bt["losses"] += 1
            n = wins + losses + pushes
            if n < min_n:
                continue
            decided = wins + losses
            wr = (wins / decided) if decided else None
            type_rows = []
            for ct, bt in sorted(by_type.items(), key=lambda x: -int(x[1]["n"])):
                d = int(bt["wins"]) + int(bt["losses"])
                type_rows.append(
                    {
                        "contract_type": ct,
                        "n": int(bt["n"]),
                        "wins": int(bt["wins"]),
                        "losses": int(bt["losses"]),
                        "pnl": round(float(bt["pnl"]), 4),
                        "win_rate": round(bt["wins"] / d, 3) if d else None,
                    }
                )
            # Compact recent trades for this bucket only (token control)
            recent = []
            for r in rows[-15:]:
                recent.append(
                    {
                        "status": r.get("status"),
                        "contract_type": r.get("contract_type"),
                        "stake": r.get("stake"),
                        "profit": r.get("profit"),
                        "confidence": r.get("confidence"),
                    }
                )
            buckets.append(
                {
                    "key": key,
                    "symbol": sym,
                    "family": fam,
                    "strategy": fam,  # alias for LLM
                    "n": n,
                    "wins": wins,
                    "losses": losses,
                    "pushes": pushes,
                    "win_rate": round(wr, 3) if wr is not None else None,
                    "pnl": round(pnl, 4),
                    "by_contract_type": type_rows,
                    "recent_trades": recent,
                }
            )
        buckets.sort(key=lambda b: (-int(b["n"]), str(b["symbol"])))
        return buckets

    def apply_recommendation(
        self,
        rec: Dict[str, Any],
        *,
        merge: bool = True,
    ) -> None:
        """
        Update type multipliers + bans/preferred from a recommendation.

        merge=True (default): only patch keys present in this analysis so
        other markets keep their prior DeepSeek weights.
        """
        self.last_recommendation = rec
        analysis = rec.get("trade_type_analysis") or []
        new_bans: set[str] = set()
        new_pref: set[str] = set()
        touched: set[str] = set()
        for row in analysis:
            if not isinstance(row, dict):
                continue
            ct = str(row.get("contract_type") or "").upper()
            sym = str(row.get("symbol") or "").strip()
            mult = row.get("suggested_confidence_mult")
            verdict = str(row.get("verdict") or "").lower()
            if mult is None:
                if verdict == "ban":
                    mult = 0.5
                elif verdict == "reduce":
                    mult = 0.85
                else:
                    mult = 1.05
            try:
                mult_f = float(mult)
            except (TypeError, ValueError):
                continue
            mult_f = max(0.5, min(1.25, mult_f))
            # Always prefer symbol|type so recs match bot trade types
            keys: List[str] = []
            if sym and ct:
                sk = f"{sym}|{ct}"
                keys.append(sk)
                self._type_multipliers[sk] = mult_f
            elif ct:
                # Type-only only if model forgot symbol — still apply lightly
                keys.append(ct)
                self._type_multipliers[ct] = mult_f
            else:
                continue
            for k in keys:
                touched.add(k)
                if verdict == "ban" or mult_f <= 0.55:
                    new_bans.add(k)
                elif verdict in {"keep", "boost", "prefer"} or mult_f >= 1.05:
                    new_pref.add(k)
        if analysis:
            if merge:
                for k in new_bans:
                    self._bans.add(k)
                    self._preferred.discard(k)
                for k in new_pref:
                    self._preferred.add(k)
                    self._bans.discard(k)
                # If verdict moved off ban for a touched key
                for k in touched:
                    if k not in new_bans and k in self._bans and k in new_pref:
                        self._bans.discard(k)
            else:
                self._bans = new_bans
                self._preferred = new_pref
        self._save_cache()
        logger.info(
            "DeepSeek applied (merge=%s): mults=%d bans=%d preferred=%d touched=%d",
            merge,
            len(self._type_multipliers),
            len(self._bans),
            len(self._preferred),
            len(touched),
        )

    def build_user_payload(
        self,
        *,
        trades: List[Dict[str, Any]],
        learning: Optional[Dict[str, Any]] = None,
        risk: Optional[Dict[str, Any]] = None,
        strategies: Optional[Dict[str, Any]] = None,
        buckets: Optional[List[Dict[str, Any]]] = None,
        scope_keys: Optional[List[str]] = None,
        force: bool = False,
    ) -> str:
        if buckets is None:
            buckets = self.build_market_strategy_buckets(trades, min_n=1)
        body = {
            "analysis_mode": "per_market_per_strategy",
            "sample_policy": {
                "min_sample_global": self.min_sample,
                "min_per_setup": self.min_per_setup,
                "analyze_every_global": self.analyze_every,
                "force": force,
            },
            "scope_keys": scope_keys or [],
            "market_strategy_buckets": buckets,
            "learning": learning or {},
            "risk_session": risk or {},
            "strategies": strategies or {},
            "goals": {
                "reduce_loss_rate": True,
                "risk_per_trade_pct": "1-2",
                "session_stop_loss_pct_band": "5-10",
                "session_target_rr": "1:3",
                "prefer_trend_following": True,
                "recommend_only_for_buckets_in_payload": True,
                "always_include_symbol_and_contract_type": True,
            },
            # Raw recent trades only as thin fallback (prefer buckets)
            "recent_trades_tail": trades[-20:],
        }
        return json.dumps(body, default=str)

    def select_buckets_for_analysis(
        self,
        trades: Sequence[Dict[str, Any]],
        *,
        force: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str], str]:
        """
        Choose which market|strategy buckets to send to DeepSeek.

        Returns (buckets, scope_keys, reason).
        Empty buckets → skip API (caller should not call).
        """
        all_buckets = self.build_market_strategy_buckets(trades, min_n=1)
        due = set(self.due_setup_keys())
        total_n = len(trades)

        if force:
            # Manual: include all buckets with at least 3 samples
            ripe = [b for b in all_buckets if int(b["n"]) >= min(3, self.min_per_setup)]
            if not ripe and all_buckets:
                ripe = all_buckets[:8]
            keys = [str(b["key"]) for b in ripe]
            return ripe, keys, "force"

        # Prefer due market|family buckets with enough absolute samples
        ripe_due = [
            b
            for b in all_buckets
            if int(b["n"]) >= self.min_per_setup
            and (str(b["key"]) in due or self.closes_since_analysis >= self.analyze_every)
        ]
        if ripe_due:
            keys = [str(b["key"]) for b in ripe_due]
            return ripe_due, keys, "per_setup_or_global"

        # Global cadence only when overall sample is large enough
        if (
            self.closes_since_analysis >= self.analyze_every
            and total_n >= self.min_sample
        ):
            ripe = [b for b in all_buckets if int(b["n"]) >= self.min_per_setup]
            if not ripe:
                ripe = [b for b in all_buckets if int(b["n"]) >= max(5, self.min_per_setup // 2)]
            keys = [str(b["key"]) for b in ripe]
            return ripe, keys, "global_cadence"

        return [], [], "insufficient_sample"

    def analyze(
        self,
        *,
        trades: List[Dict[str, Any]],
        learning: Optional[Dict[str, Any]] = None,
        risk: Optional[Dict[str, Any]] = None,
        strategies: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Call DeepSeek and return parsed recommendation dict, or None on failure.

        force=True: dashboard/manual (still prefers grouped buckets).
        force=False: requires adequate sample (min_sample or ripe per-setup).
        """
        if not self.is_ready():
            self.last_error = "deepseek_disabled_or_no_key"
            logger.info("DeepSeek analyze skipped: %s", self.last_error)
            return None

        buckets, scope_keys, reason = self.select_buckets_for_analysis(
            trades, force=force
        )
        if not buckets:
            self.last_error = (
                f"insufficient_sample (need ≥{self.min_sample} closes overall "
                f"or ≥{self.min_per_setup} on a market|strategy; have {len(trades)})"
            )
            logger.info("DeepSeek analyze skipped: %s", self.last_error)
            return None

        if not force and len(trades) < self.min_sample and reason != "per_setup_or_global":
            # Per-setup path already filtered by min_per_setup in select
            pass

        # Drop thin buckets when not force
        if not force:
            buckets = [
                b for b in buckets if int(b["n"]) >= self.min_per_setup
            ] or buckets
            if not buckets:
                self.last_error = f"insufficient_per_setup_sample (min={self.min_per_setup})"
                logger.info("DeepSeek analyze skipped: %s", self.last_error)
                return None

        user_content = self.build_user_payload(
            trades=list(trades),
            learning=learning,
            risk=risk,
            strategies=strategies,
            buckets=buckets,
            scope_keys=scope_keys,
            force=force,
        )
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Analyze these per-market / per-strategy buckets and return "
                        "JSON only (no markdown fences). Only recommend for symbols "
                        "and contract types present in market_strategy_buckets:\n"
                        + user_content
                    ),
                },
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }

        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                self.last_error = f"http_{resp.status_code}: {resp.text[:300]}"
                logger.warning("DeepSeek API error: %s", self.last_error)
                return None
            data = resp.json()
            content = (
                (data.get("choices") or [{}])[0]
                .get("message", {})
                .get("content")
                or ""
            )
            rec = self._parse_json_content(content)
            if not rec:
                self.last_error = "invalid_json_response"
                logger.warning("DeepSeek returned non-JSON: %s", content[:200])
                return None
            rec["_meta"] = {
                "model": self.model,
                "analyzed_at": time.time(),
                "n_trades": len(trades),
                "n_buckets": len(buckets),
                "scope_keys": scope_keys,
                "trigger": reason,
                "force": force,
            }
            self.last_error = None
            self.apply_recommendation(rec, merge=True)
            self.mark_analyzed(scope_keys if scope_keys else None)
            logger.info(
                "DeepSeek recommendation: score=%s buckets=%s trigger=%s summary=%s",
                rec.get("risk_score"),
                len(buckets),
                reason,
                str(rec.get("summary") or "")[:120],
            )
            return rec
        except Exception as e:
            self.last_error = str(e)
            logger.warning("DeepSeek analyze failed: %s", e)
            return None

    @staticmethod
    def _parse_json_content(content: str) -> Optional[Dict[str, Any]]:
        text = (content or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(text[start : end + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
            return None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.is_ready(),
            "model": self.model,
            "analyze_every": self.analyze_every,
            "min_sample": self.min_sample,
            "min_per_setup": self.min_per_setup,
            "closes_since_analysis": self.closes_since_analysis,
            "due_setups": self.due_setup_keys(),
            "setup_closes": dict(self._setup_closes),
            "last_error": self.last_error,
            "type_multipliers": dict(self._type_multipliers),
            "bans": sorted(self._bans),
            "preferred": sorted(self._preferred),
            "preferred_symbols": self.preferred_symbols(),
            "recommendation": self.last_recommendation,
            "skill_path": str(self.skill_path),
            "configured": bool(self.api_key),
            "key_prefix": (self.api_key[:6] + "…") if self.api_key else None,
        }
