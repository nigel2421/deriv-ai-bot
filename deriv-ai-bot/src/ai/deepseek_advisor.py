"""
DeepSeek advisor — analyzes trade runs and recommends strategy improvements.

Uses the OpenAI-compatible DeepSeek chat API (httpx, no openai package required).
Recommendations feed the adaptive learner and operator dashboard.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_SKILL_PATH = Path(__file__).resolve().parents[2] / "skills" / "deepseek-trading" / "SKILL.md"
DEFAULT_CACHE_PATH = Path("data/deepseek_recommendations.json")

# Fallback system prompt if SKILL.md is missing
_FALLBACK_SYSTEM = """You are the DeepSeek trading advisor for a Deriv synthetic-indices bot
(Volatility 10/25/50/75/100 + 1Hz). Protect capital first.

Rules:
- Risk 1–2% of balance per trade max.
- Session stop-loss: 5–10% of session-start balance (operator-set).
- Session profit target: 1:3 risk:reward (target = stop_loss_amount × 3).
- Prefer trend-following: 50/200 EMA stack, RSI confirmation, pullback entries.
- Vol 75/100: 15m trend, 5m entry; never chase large candles; wait for retrace.
- Boom: after spike-down + bullish confirm. Crash: after spike-up + bearish confirm.
- Trade types must be analyzed individually (CALL/PUT vs DIGITOVER/UNDER/EVEN/ODD).
- Only high-quality setups; skip chop; require structure/EMA/RSI agreement.

Respond with JSON only:
{
  "summary": "short overall assessment",
  "risk_score": 0-100,
  "trade_type_analysis": [{"contract_type": "...", "symbol": "...", "verdict": "keep|reduce|ban", "reason": "...", "suggested_confidence_mult": 0.5-1.2}],
  "strategy_changes": ["..."],
  "stake_advice": {"action": "keep|lower|raise", "pct_of_balance": 1.0, "reason": "..."},
  "session_advice": {"stop_loss_pct": 5.0, "target_rr": 3.0, "reason": "..."},
  "learning_hints": ["..."]
}
"""


def load_skill_prompt(path: Optional[Path] = None) -> str:
    p = Path(path) if path else DEFAULT_SKILL_PATH
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Could not read DeepSeek skill file %s: %s", p, e)
    return _FALLBACK_SYSTEM


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
        analyze_every: int = 5,
    ):
        raw_key = (api_key or "").strip() or None
        self.api_key = raw_key
        self.base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self.model = model or "deepseek-v4-flash"
        self.timeout_sec = float(timeout_sec)
        self.skill_path = Path(skill_path) if skill_path else DEFAULT_SKILL_PATH
        self.cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
        self.analyze_every = max(0, int(analyze_every))
        self.system_prompt = load_skill_prompt(self.skill_path)
        self.last_recommendation: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self.closes_since_analysis = 0
        self._type_multipliers: Dict[str, float] = {}
        # Validate key format: DeepSeek keys are sk-…  Google AIzaSy… is wrong secret
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
                # Allow other formats but warn
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
            logger.info(
                "DeepSeekAdvisor loaded cache (%d type mults) from %s",
                len(self._type_multipliers),
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
                "updated_at": time.time(),
            }
            self.cache_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.debug("DeepSeek cache save failed: %s", e)

    def is_ready(self) -> bool:
        return bool(self.enabled and self.api_key)

    def note_closed_trade(self) -> bool:
        """
        Increment close counter. Returns True if an analysis should run now.
        """
        if not self.is_ready() or self.analyze_every <= 0:
            return False
        self.closes_since_analysis += 1
        if self.closes_since_analysis >= self.analyze_every:
            self.closes_since_analysis = 0
            return True
        return False

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

    def apply_recommendation(self, rec: Dict[str, Any]) -> None:
        """Update type multipliers from a recommendation payload."""
        self.last_recommendation = rec
        analysis = rec.get("trade_type_analysis") or []
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
            if ct:
                self._type_multipliers[ct] = mult_f
            if sym and ct:
                self._type_multipliers[f"{sym}|{ct}"] = mult_f
        self._save_cache()

    def build_user_payload(
        self,
        *,
        trades: List[Dict[str, Any]],
        learning: Optional[Dict[str, Any]] = None,
        risk: Optional[Dict[str, Any]] = None,
        strategies: Optional[Dict[str, Any]] = None,
    ) -> str:
        body = {
            "recent_trades": trades[-40:],
            "learning": learning or {},
            "risk_session": risk or {},
            "strategies": strategies or {},
            "goals": {
                "reduce_loss_rate": True,
                "risk_per_trade_pct": "1-2",
                "session_stop_loss_pct_band": "5-10",
                "session_target_rr": "1:3",
                "prefer_trend_following": True,
                "multi_tf_vol75_100": "15m trend / 5m entry",
            },
        }
        return json.dumps(body, default=str)

    def analyze(
        self,
        *,
        trades: List[Dict[str, Any]],
        learning: Optional[Dict[str, Any]] = None,
        risk: Optional[Dict[str, Any]] = None,
        strategies: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Call DeepSeek and return parsed recommendation dict, or None on failure.
        """
        if not self.is_ready():
            self.last_error = "deepseek_disabled_or_no_key"
            logger.info("DeepSeek analyze skipped: %s", self.last_error)
            return None

        user_content = self.build_user_payload(
            trades=trades,
            learning=learning,
            risk=risk,
            strategies=strategies,
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
                        "Analyze this bot run and return JSON only "
                        "(no markdown fences):\n" + user_content
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
            }
            self.last_error = None
            self.apply_recommendation(rec)
            logger.info(
                "DeepSeek recommendation: risk_score=%s summary=%s",
                rec.get("risk_score"),
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
        # Strip optional markdown fences
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
            # Try to find outermost {...}
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
            "closes_since_analysis": self.closes_since_analysis,
            "last_error": self.last_error,
            "type_multipliers": dict(self._type_multipliers),
            "recommendation": self.last_recommendation,
            "skill_path": str(self.skill_path),
        }
