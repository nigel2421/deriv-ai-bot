"""
Market category taxonomy for multi-asset Deriv scanning.

Each class gets its own scoring engine. Synthetic sub-types
(Step, DSI, Jump, DEX, Trek, etc.) use specialized metric stacks.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

FOREX = "forex"
STOCKS = "stocks"
INDICES = "indices"
COMMODITIES = "commodities"
CRYPTO = "crypto"
SYNTHETIC_VOL = "synthetic_vol"
BOOM = "boom"
CRASH = "crash"
STEP = "step"
DSI = "dsi"  # drift switching
VOL_SWITCH = "vol_switch"
JUMP = "jump"
DEX = "dex"
TREK = "trek"
SKEW_STEP = "skew_step"
DAILY_RESET = "daily_reset"
DERIVED_FX = "derived_fx"
UNKNOWN = "unknown"

ALL_CATEGORIES = (
    FOREX,
    STOCKS,
    INDICES,
    COMMODITIES,
    CRYPTO,
    SYNTHETIC_VOL,
    BOOM,
    CRASH,
    STEP,
    DSI,
    VOL_SWITCH,
    JUMP,
    DEX,
    TREK,
    SKEW_STEP,
    DAILY_RESET,
    DERIVED_FX,
    UNKNOWN,
)

# ---------------------------------------------------------------------------
# Scoring engines
# ---------------------------------------------------------------------------

ENGINE_FOREX = {
    "momentum": 0.25,
    "persistence": 0.20,
    "transition": 0.15,
    "volatility_regime": 0.15,
    "hpp": 0.15,
    "hpp_velocity": 0.10,
}
ENGINE_STOCKS = {
    "momentum": 0.30,
    "persistence": 0.25,
    "trend_strength": 0.25,
    "hpp": 0.20,
}
ENGINE_INDICES = {
    "momentum": 0.30,
    "persistence": 0.25,
    "trend_strength": 0.25,
    "directional_entropy": 0.20,
}
ENGINE_COMMODITIES = {
    "momentum": 0.35,
    "volatility_regime": 0.25,
    "persistence": 0.25,
    "trend_strength": 0.15,
}
ENGINE_CRYPTO = {
    "momentum": 0.25,
    "persistence": 0.20,
    "acceleration": 0.20,
    "volatility_regime": 0.20,
    "hpp": 0.15,
}
ENGINE_SYNTHETIC_VOL = {
    "entropy": 0.20,
    "pattern_strength": 0.20,
    "pattern_clarity": 0.20,
    "momentum": 0.20,
    "persistence": 0.20,
}
ENGINE_BOOM = {
    "spike_analysis": 0.40,
    "persistence": 0.25,
    "pattern_strength": 0.20,
    "momentum": 0.15,
}
ENGINE_CRASH = {
    "spike_analysis": 0.40,
    "persistence": 0.25,
    "pattern_strength": 0.20,
    "momentum": 0.15,
}
# Step indices — fixed increments
ENGINE_STEP = {
    "momentum": 0.35,
    "persistence": 0.35,
    "transition": 0.30,
}
# Drift switching
ENGINE_DSI = {
    "regime_detection": 0.40,
    "persistence": 0.30,
    "momentum": 0.30,
}
# Volatility switch
ENGINE_VOL_SWITCH = {
    "regime_classification": 0.50,
    "volatility_engine": 0.50,
}
# Jump indices
ENGINE_JUMP = {
    "jump_probability": 0.40,
    "entropy": 0.30,
    "persistence": 0.30,
}
# DEX simulated news spikes
ENGINE_DEX = {
    "spike_prediction": 0.40,
    "momentum": 0.30,
    "persistence": 0.30,
}
# Trek — directional bias
ENGINE_TREK = {
    "direction_persistence": 0.50,
    "trend_strength": 0.50,
}
# Skew step
ENGINE_SKEW_STEP = {
    "bias_detection": 0.40,
    "entropy": 0.30,
    "persistence": 0.30,
}
# Daily reset bull/bear
ENGINE_DAILY_RESET = {
    "trend_following": 0.50,
    "momentum": 0.50,
}
# Derived FX (smoothed)
ENGINE_DERIVED_FX = {
    "momentum": 0.35,
    "persistence": 0.35,
    "regime_detection": 0.30,
}

CATEGORY_ENGINES: Dict[str, Dict[str, float]] = {
    FOREX: ENGINE_FOREX,
    STOCKS: ENGINE_STOCKS,
    INDICES: ENGINE_INDICES,
    COMMODITIES: ENGINE_COMMODITIES,
    CRYPTO: ENGINE_CRYPTO,
    SYNTHETIC_VOL: ENGINE_SYNTHETIC_VOL,
    BOOM: ENGINE_BOOM,
    CRASH: ENGINE_CRASH,
    STEP: ENGINE_STEP,
    DSI: ENGINE_DSI,
    VOL_SWITCH: ENGINE_VOL_SWITCH,
    JUMP: ENGINE_JUMP,
    DEX: ENGINE_DEX,
    TREK: ENGINE_TREK,
    SKEW_STEP: ENGINE_SKEW_STEP,
    DAILY_RESET: ENGINE_DAILY_RESET,
    DERIVED_FX: ENGINE_DERIVED_FX,
    UNKNOWN: ENGINE_SYNTHETIC_VOL,
}

DIGIT_TYPES = frozenset(
    {
        "DIGITOVER",
        "DIGITUNDER",
        "DIGITEVEN",
        "DIGITODD",
        "DIGITDIFF",
        "DIGITMATCH",
    }
)
RF_TYPES = frozenset({"CALL", "PUT"})

# Digits only where last-digit stream is meaningful (classic vol + some jumps)
CATEGORY_CONTRACTS: Dict[str, Set[str]] = {
    FOREX: set(RF_TYPES),
    STOCKS: set(RF_TYPES),
    INDICES: set(RF_TYPES),
    COMMODITIES: set(RF_TYPES),
    CRYPTO: set(RF_TYPES),
    SYNTHETIC_VOL: set(DIGIT_TYPES) | set(RF_TYPES),
    BOOM: set(RF_TYPES),
    CRASH: set(RF_TYPES),
    STEP: set(RF_TYPES),
    DSI: set(RF_TYPES),
    VOL_SWITCH: set(RF_TYPES),
    JUMP: set(DIGIT_TYPES) | set(RF_TYPES),
    DEX: set(RF_TYPES),
    TREK: set(RF_TYPES),
    SKEW_STEP: set(RF_TYPES),
    DAILY_RESET: set(RF_TYPES),
    DERIVED_FX: set(RF_TYPES),
    UNKNOWN: set(RF_TYPES),
}

CATEGORY_EXAMPLES: Dict[str, List[str]] = {
    FOREX: ["frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxUSDCHF", "frxAUDUSD"],
    STOCKS: ["Apple", "Tesla", "NVIDIA", "Meta", "Microsoft"],
    INDICES: ["US 500", "US Tech 100", "Wall Street 30", "FTSE 100", "DAX"],
    COMMODITIES: ["Gold", "Silver", "Crude Oil", "Natural Gas"],
    CRYPTO: ["cryBTCUSD", "cryETHUSD", "cryLTCUSD", "cryXRPUSD"],
    SYNTHETIC_VOL: [
        "R_10",
        "R_25",
        "R_50",
        "R_75",
        "R_100",
        "1HZ10V",
        "1HZ25V",
        "1HZ50V",
        "1HZ75V",
        "1HZ100V",
    ],
    BOOM: ["BOOM300N", "BOOM500", "BOOM600", "BOOM900", "BOOM1000"],
    CRASH: ["CRASH300N", "CRASH500", "CRASH600", "CRASH900", "CRASH1000"],
    STEP: ["stpRNG", "stpRNGzx", "Step 0.1", "Step 0.2", "Step 0.5"],
    DSI: ["DSI10", "DSI20", "DSI30"],
    VOL_SWITCH: ["WLDXAU", "Vol Switch"],  # symbols vary by account
    JUMP: ["JD10", "JD25", "JD50", "JD75", "JD100", "R_10", "Jump 50"],
    DEX: ["DEX600UP", "DEX600DN", "DEX900UP", "DEX900DN", "DEX1500UP", "DEX1500DN"],
    TREK: ["TREK", "Trek Index"],
    SKEW_STEP: ["Skew Step"],
    DAILY_RESET: ["RDBULL", "RDBEAR"],
    DERIVED_FX: [
        "frxGBPUSDDFx10",
        "frxGBPUSDDFx20",
        "frxEURUSDDFx10",
        "frxEURUSDDFx20",
        "frxAUDUSDDFx10",
        "frxUSDJPYDFx20",
    ],
}

PATH_DIGITS_RF = "digits_and_rf"
PATH_DIRECTIONAL = "directional"
PATH_SPIKE = "spike"
PATH_REGIME = "regime"
PATH_JUMP = "jump"

_FOREX_PREFIXES = ("frx", "fx_", "forex")
_CRYPTO_PREFIXES = ("cry", "crypto")
_STOCK_HINTS = ("aapl", "tsla", "nvda", "meta", "msft", "amzn", "googl", "stock")
_INDEX_HINTS = (
    "otc_",
    "us500",
    "ustec",
    "ws30",
    "ftse",
    "gdaxi",
    "n225",
    "asx",
    "hsi",
    "spc",
    "dji",
)
_COMMODITY_HINTS = (
    "frxxau",
    "frxxag",
    "frxxbr",
    "frxxng",
    "wti",
    "brent",
    "gold",
    "silver",
    "oil",
    "sugar",
    "natgas",
)


def normalize_symbol(symbol: Optional[str]) -> str:
    return str(symbol or "").strip()


def classify_market(symbol: str) -> str:
    """Map a Deriv symbol to a market category (most specific first)."""
    s = normalize_symbol(symbol)
    if not s:
        return UNKNOWN
    u = s.upper()
    low = s.lower()

    # Derived FX (smoothed forex) before plain forex
    if "DFX" in u or "DFX" in u.replace("_", ""):
        return DERIVED_FX
    if re.search(r"DFx\d+", s, re.I):
        return DERIVED_FX

    # Boom / Crash
    if "BOOM" in u:
        return BOOM
    if "CRASH" in u:
        return CRASH

    # DEX
    if u.startswith("DEX") or "DEX" in u and any(x in u for x in ("UP", "DN", "DOWN")):
        return DEX

    # DSI drift switching
    if re.match(r"^DSI\d+", u) or "DRIFT" in u:
        return DSI

    # Jump indices
    if re.match(r"^JD\d+", u) or re.match(r"^JUMP", u) or "JUMP" in u:
        return JUMP

    # Step / Skew step
    if "SKEW" in u and "STEP" in u:
        return SKEW_STEP
    if re.match(r"^STP", u) or "STEP" in u or low.startswith("stprng"):
        return STEP

    # Trek
    if "TREK" in u:
        return TREK

    # Daily reset bull/bear
    if u.startswith("RD") and any(x in u for x in ("BULL", "BEAR")):
        return DAILY_RESET
    if "DAILYRESET" in u or "RDBULL" in u or "RDBEAR" in u:
        return DAILY_RESET

    # Volatility switch
    if "VOLSWITCH" in u or "VSI" in u or ("SWITCH" in u and "VOL" in u):
        return VOL_SWITCH

    # Classic synthetic volatility
    if re.match(r"^R_\d+", u) or re.match(r"^1HZ\d+V?$", u) or "VOLATILITY" in u:
        return SYNTHETIC_VOL

    # Forex
    if any(low.startswith(p) for p in _FOREX_PREFIXES):
        if any(h in low for h in ("xau", "xag", "xbr", "xng", "wti")):
            return COMMODITIES
        return FOREX
    if re.match(r"^[A-Z]{3}/?[A-Z]{3}$", u.replace("_", "")):
        return FOREX

    # Crypto
    if any(low.startswith(p) for p in _CRYPTO_PREFIXES):
        return CRYPTO
    if any(x in u for x in ("BTC", "ETH", "LTC", "XRP", "BCH", "DOGE")):
        if not any(x in u for x in ("R_", "1HZ")):
            return CRYPTO

    if any(h in low for h in _COMMODITY_HINTS):
        return COMMODITIES
    if any(h in low for h in _INDEX_HINTS):
        return INDICES
    if any(h in low for h in _STOCK_HINTS):
        return STOCKS

    return UNKNOWN


def scoring_engine(category: str) -> Dict[str, float]:
    eng = dict(CATEGORY_ENGINES.get(category) or ENGINE_SYNTHETIC_VOL)
    s = sum(eng.values()) or 1.0
    return {k: v / s for k, v in eng.items()}


def scoring_path(category: str) -> str:
    if category == SYNTHETIC_VOL:
        return PATH_DIGITS_RF
    if category in {BOOM, CRASH, DEX}:
        return PATH_SPIKE
    if category in {DSI, VOL_SWITCH}:
        return PATH_REGIME
    if category == JUMP:
        return PATH_JUMP
    return PATH_DIRECTIONAL


def allowed_contracts(category: str) -> Set[str]:
    return set(CATEGORY_CONTRACTS.get(category) or RF_TYPES)


def filter_allowed_for_symbol(
    symbol: str,
    candidate_types: Sequence[str],
) -> List[str]:
    cat = classify_market(symbol)
    allowed = allowed_contracts(cat)
    return [str(t).upper() for t in candidate_types if str(t or "").upper() in allowed]


def market_profile(symbol: str) -> Dict[str, Any]:
    cat = classify_market(symbol)
    eng = scoring_engine(cat)
    return {
        "symbol": normalize_symbol(symbol),
        "category": cat,
        "label": cat.replace("_", " ").title(),
        "scoring_path": scoring_path(cat),
        "engine_weights": eng,
        "primary_metrics": list(eng.keys()),
        "allowed_contracts": sorted(allowed_contracts(cat)),
        "digits_enabled": bool(allowed_contracts(cat) & DIGIT_TYPES),
        "rf_enabled": bool(allowed_contracts(cat) & RF_TYPES),
        "examples": CATEGORY_EXAMPLES.get(cat) or [],
    }


def profiles_for_symbols(symbols: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    return {s: market_profile(s) for s in symbols if s}


def category_summary() -> List[Dict[str, Any]]:
    rows = []
    for cat in ALL_CATEGORIES:
        if cat == UNKNOWN:
            continue
        eng = scoring_engine(cat)
        rows.append(
            {
                "category": cat,
                "label": cat.replace("_", " ").title(),
                "path": scoring_path(cat),
                "metrics": ", ".join(eng.keys()),
                "contracts": ", ".join(sorted(allowed_contracts(cat))),
                "examples": ", ".join((CATEGORY_EXAMPLES.get(cat) or [])[:5]),
            }
        )
    return rows


# Opt-in symbol books
PRESET_SYNTHETIC_FOCUS = "R_25,R_50,1HZ50V,R_10"
PRESET_SYNTHETIC_FULL = (
    "R_10,R_25,R_50,R_75,R_100,1HZ10V,1HZ25V,1HZ50V,1HZ75V,1HZ100V"
)
PRESET_BOOM_CRASH = "BOOM500,BOOM1000,CRASH500,CRASH1000"
PRESET_FOREX_MAJORS = (
    "frxEURUSD,frxGBPUSD,frxUSDJPY,frxUSDCHF,frxAUDUSD,frxNZDUSD,frxUSDCAD"
)
PRESET_CRYPTO = "cryBTCUSD,cryETHUSD"
PRESET_JUMP = "JD10,JD25,JD50,JD75,JD100"
PRESET_DSI = "DSI10,DSI20,DSI30"
