"""Unified contract type helpers: digits + Rise/Fall (CALL/PUT)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

from src.strategy.digit_contracts import (
    ALL_DIGIT_TYPES,
    BARRIER_TYPES,
    NO_BARRIER_TYPES,
    build_proposal_fields as build_digit_proposal_fields,
    normalize_contract_type as normalize_digit_type,
    validate_digit_contract,
)

# Deriv Rise/Fall
RISE_FALL_TYPES = frozenset({"CALL", "PUT"})
# Aliases sometimes used in UI
_ALIASES = {
    "RISE": "CALL",
    "FALL": "PUT",
    "HIGHER": "CALL",
    "LOWER": "PUT",
}

ALL_TRADE_TYPES = ALL_DIGIT_TYPES | RISE_FALL_TYPES


def normalize_contract_type(contract_type: Optional[str]) -> Optional[str]:
    if not contract_type:
        return None
    ct = str(contract_type).strip().upper()
    ct = _ALIASES.get(ct, ct)
    if ct in ALL_TRADE_TYPES:
        return ct
    # Digit normalizer for anything else known
    return normalize_digit_type(ct)


def is_digit_contract(contract_type: Optional[str]) -> bool:
    ct = normalize_contract_type(contract_type)
    return ct is not None and ct in ALL_DIGIT_TYPES


def is_rise_fall(contract_type: Optional[str]) -> bool:
    ct = normalize_contract_type(contract_type)
    return ct is not None and ct in RISE_FALL_TYPES


def requires_barrier(contract_type: Optional[str]) -> bool:
    ct = normalize_contract_type(contract_type)
    return ct is not None and ct in BARRIER_TYPES


def validate_contract(
    contract_type: str,
    barrier: Optional[int] = None,
) -> Tuple[bool, str, Optional[int]]:
    ct = normalize_contract_type(contract_type)
    if ct is None:
        return False, f"unsupported_contract_type:{contract_type}", None
    if ct in RISE_FALL_TYPES:
        return True, "ok", None
    if ct in NO_BARRIER_TYPES:
        return True, "ok", None
    return validate_digit_contract(ct, barrier)


def build_proposal_fields(
    contract_type: str,
    barrier: Optional[int] = None,
) -> Dict[str, Any]:
    ct = normalize_contract_type(contract_type)
    if ct is None:
        raise ValueError(f"unsupported_contract_type:{contract_type}")
    if ct in RISE_FALL_TYPES:
        return {"contract_type": ct}
    return build_digit_proposal_fields(ct, barrier)
