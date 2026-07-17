"""
Deriv digit-contract helpers.

Rules (last digit of settlement tick, 0–9):
  DIGITOVER  — win if digit > barrier   (barrier 0–8; barrier 9 never wins)
  DIGITUNDER — win if digit < barrier   (barrier 1–9; barrier 0 never wins)
  DIGITMATCH — win if digit == barrier  (barrier 0–9)
  DIGITDIFF  — win if digit != barrier  (barrier 0–9)
  DIGITEVEN  — win if digit in {0,2,4,6,8}  (no barrier)
  DIGITODD   — win if digit in {1,3,5,7,9}  (no barrier)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Contract families
BARRIER_TYPES = frozenset(
    {"DIGITOVER", "DIGITUNDER", "DIGITMATCH", "DIGITDIFF"}
)
NO_BARRIER_TYPES = frozenset({"DIGITEVEN", "DIGITODD"})
ALL_DIGIT_TYPES = BARRIER_TYPES | NO_BARRIER_TYPES

# Strategy defaults: OVER 6 (win 7–9), UNDER 4 (win 0–3)
DEFAULT_OVER_BARRIER = 6   # win if digit > 6 → 7,8,9
DEFAULT_UNDER_BARRIER = 4  # win if digit < 4 → 0,1,2,3


def extract_last_digit(quote: Any) -> Optional[int]:
    """
    Extract the last digit of a price quote the way digit contracts use it.

    Uses the final numeric character of the quote string so that e.g.
    503.77 → 7, 54640.6196 → 6, 100 → 0.
    """
    if quote is None:
        return None
    try:
        # Prefer string form to avoid float binary noise
        if isinstance(quote, float):
            # Format without scientific notation; strip trailing zeros carefully
            s = f"{quote:.10f}".rstrip("0").rstrip(".")
        else:
            s = str(quote).strip()
        digits = re.findall(r"\d", s)
        if not digits:
            return None
        return int(digits[-1])
    except (TypeError, ValueError):
        return None


def last_digits_from_ticks(ticks: Sequence[Dict[str, Any]], n: int = 20) -> List[int]:
    """Last digits from the most recent n ticks (oldest→newest)."""
    out: List[int] = []
    for t in list(ticks)[-n:]:
        d = extract_last_digit(t.get("quote") if isinstance(t, dict) else t)
        if d is not None:
            out.append(d)
    return out


def requires_barrier(contract_type: str) -> bool:
    return str(contract_type).upper() in BARRIER_TYPES


def normalize_contract_type(contract_type: Optional[str]) -> Optional[str]:
    if not contract_type:
        return None
    ct = str(contract_type).strip().upper()
    if ct not in ALL_DIGIT_TYPES:
        return None
    return ct


def clamp_digit(value: Any, default: int = 5) -> int:
    try:
        d = int(value)
    except (TypeError, ValueError):
        d = default
    return max(0, min(9, d))


def normalize_barrier(
    contract_type: str,
    barrier: Optional[int],
    *,
    default_over: int = DEFAULT_OVER_BARRIER,
    default_under: int = DEFAULT_UNDER_BARRIER,
) -> Optional[int]:
    """
    Return a valid barrier for the contract type, or None if not required.
    Adjusts impossible edges (OVER@9 → 8, UNDER@0 → 1).
    """
    ct = normalize_contract_type(contract_type)
    if ct is None or ct in NO_BARRIER_TYPES:
        return None

    if barrier is None:
        if ct == "DIGITOVER":
            b = default_over
        elif ct == "DIGITUNDER":
            b = default_under
        else:
            b = 5  # MATCH/DIFF default middle
    else:
        b = clamp_digit(barrier)

    if ct == "DIGITOVER":
        # digit > b must be possible → b in 0..8
        b = min(b, 8)
    elif ct == "DIGITUNDER":
        # digit < b must be possible → b in 1..9
        b = max(b, 1)

    return b


def validate_digit_contract(
    contract_type: str,
    barrier: Optional[int] = None,
) -> Tuple[bool, str, Optional[int]]:
    """
    Validate type + barrier for a Deriv digit proposal.

    Returns (ok, reason, normalized_barrier).
    """
    ct = normalize_contract_type(contract_type)
    if ct is None:
        return False, f"unsupported_contract_type:{contract_type}", None

    if ct in NO_BARRIER_TYPES:
        return True, "ok", None

    nb = normalize_barrier(ct, barrier)
    if nb is None:
        return False, "barrier_required", None

    if ct == "DIGITOVER" and nb >= 9:
        return False, "DIGITOVER barrier must be 0-8", None
    if ct == "DIGITUNDER" and nb <= 0:
        return False, "DIGITUNDER barrier must be 1-9", None

    return True, "ok", nb


def barrier_for_predicted_digit(
    contract_type: str,
    predicted_digit: int,
) -> Optional[int]:
    """
    Choose a barrier consistent with a predicted last digit.

    OVER:  barrier = predicted - 1  (need digit > barrier; pred is in win set if pred > barrier)
    UNDER: barrier = predicted + 1
    MATCH/DIFF: barrier = predicted
    EVEN/ODD: None
    """
    ct = normalize_contract_type(contract_type)
    if ct is None or ct in NO_BARRIER_TYPES:
        return None

    d = clamp_digit(predicted_digit)

    if ct == "DIGITOVER":
        # Prefer barrier just below prediction; floor 0, cap 8
        return normalize_barrier(ct, max(0, d - 1))
    if ct == "DIGITUNDER":
        return normalize_barrier(ct, min(9, d + 1))
    if ct in {"DIGITMATCH", "DIGITDIFF"}:
        return d
    return normalize_barrier(ct, d)


def would_win(contract_type: str, barrier: Optional[int], digit: int) -> bool:
    """Whether a settled last digit would win the contract."""
    ct = normalize_contract_type(contract_type)
    d = clamp_digit(digit)
    if ct == "DIGITOVER":
        b = normalize_barrier(ct, barrier)
        return b is not None and d > b
    if ct == "DIGITUNDER":
        b = normalize_barrier(ct, barrier)
        return b is not None and d < b
    if ct == "DIGITMATCH":
        b = normalize_barrier(ct, barrier)
        return b is not None and d == b
    if ct == "DIGITDIFF":
        b = normalize_barrier(ct, barrier)
        return b is not None and d != b
    if ct == "DIGITEVEN":
        return d % 2 == 0
    if ct == "DIGITODD":
        return d % 2 == 1
    return False


def build_proposal_fields(
    contract_type: str,
    barrier: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fields to merge into a Deriv proposal payload.
    Raises ValueError if invalid.
    """
    ok, reason, nb = validate_digit_contract(contract_type, barrier)
    if not ok:
        raise ValueError(reason)
    ct = normalize_contract_type(contract_type)
    fields: Dict[str, Any] = {"contract_type": ct}
    if nb is not None:
        fields["barrier"] = str(int(nb))
    return fields
