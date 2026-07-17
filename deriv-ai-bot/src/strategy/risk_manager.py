import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    """Result of a pre-trade risk check."""

    allowed: bool
    reason: str = "ok"

    def __bool__(self) -> bool:
        return self.allowed


class RiskManager:
    """
    Centralized risk management.

    Tracks daily PnL, consecutive losses, pauses, stake limits, and open-trade caps.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 5.0,
        max_consecutive_losses: int = 6,
        trade_pause_minutes: int = 60,
        min_balance: float = 1.0,
        max_open_trades: int = 3,
        max_stake_pct: float = 5.0,
        min_stake: float = 0.35,
        max_stake: Optional[float] = None,
    ):
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.max_consecutive_losses = int(max_consecutive_losses)
        self.trade_pause_minutes = int(trade_pause_minutes)
        self.min_balance = float(min_balance)
        self.max_open_trades = int(max_open_trades)
        self.max_stake_pct = float(max_stake_pct)
        self.min_stake = float(min_stake)
        self.max_stake = float(max_stake) if max_stake is not None else None

        self.daily_pnl: float = 0.0  # net PnL for the calendar day
        self.consecutive_losses: int = 0
        self.trades_today: int = 0
        self.wins_today: int = 0
        self.losses_today: int = 0
        self.last_reset_date = datetime.now().date()
        self.paused_until: Optional[datetime] = None
        self.session_start_balance: Optional[float] = None
        self.last_known_balance: Optional[float] = None

    def set_session_balance(self, balance: float) -> None:
        """Record starting balance once per session (or after reconnect)."""
        self.session_start_balance = float(balance)
        self.last_known_balance = float(balance)
        logger.info("Risk session balance set to %.2f", balance)

    def update_balance(self, balance: float) -> None:
        self.last_known_balance = float(balance)

    def _maybe_reset_daily(self) -> None:
        today = datetime.now().date()
        if today != self.last_reset_date:
            logger.info(
                "New trading day — resetting daily risk counters (prev pnl=%.2f)",
                self.daily_pnl,
            )
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.wins_today = 0
            self.losses_today = 0
            self.last_reset_date = today
            # Do not clear consecutive_losses across midnight by default;
            # only clear pause if expired (checked elsewhere).

    def is_paused(self) -> bool:
        if self.paused_until and datetime.now() < self.paused_until:
            return True
        if self.paused_until and datetime.now() >= self.paused_until:
            self.paused_until = None
        return False

    def pause(self, minutes: Optional[int] = None, reason: str = "") -> None:
        mins = minutes if minutes is not None else self.trade_pause_minutes
        self.paused_until = datetime.now() + timedelta(minutes=mins)
        logger.warning(
            "Trading paused for %s minutes%s",
            mins,
            f" ({reason})" if reason else "",
        )

    def resume(self) -> None:
        self.paused_until = None
        logger.info("Trading pause cleared.")

    def daily_loss_amount(self) -> float:
        """Positive number = money lost today (net)."""
        return max(0.0, -self.daily_pnl)

    def max_daily_loss_amount(self, account_balance: float) -> float:
        ref = self.session_start_balance if self.session_start_balance else account_balance
        return max(0.0, ref * (self.max_daily_loss_pct / 100.0))

    def can_trade(
        self,
        account_balance: Optional[float],
        open_trades: int = 0,
        proposed_stake: Optional[float] = None,
    ) -> RiskDecision:
        """
        Pre-trade risk checks.

        account_balance: live balance (None = unknown → deny).
        open_trades: currently open contracts.
        proposed_stake: optional stake to validate affordability / size.
        """
        self._maybe_reset_daily()

        if account_balance is None:
            return RiskDecision(False, "balance_unknown")

        try:
            balance = float(account_balance)
        except (TypeError, ValueError):
            return RiskDecision(False, "balance_invalid")

        self.last_known_balance = balance
        if self.session_start_balance is None:
            self.set_session_balance(balance)

        if self.is_paused():
            remaining = (self.paused_until - datetime.now()).total_seconds() / 60.0  # type: ignore[operator]
            return RiskDecision(False, f"paused ({remaining:.0f}m left)")

        if balance < self.min_balance:
            return RiskDecision(
                False,
                f"balance_below_min ({balance:.2f} < {self.min_balance:.2f})",
            )

        daily_max = self.max_daily_loss_amount(balance)
        lost = self.daily_loss_amount()
        if daily_max > 0 and lost >= daily_max:
            return RiskDecision(
                False,
                f"daily_loss_limit ({lost:.2f} >= {daily_max:.2f})",
            )

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.pause(reason=f"{self.consecutive_losses} consecutive losses")
            return RiskDecision(
                False,
                f"max_consecutive_losses ({self.consecutive_losses})",
            )

        if open_trades >= self.max_open_trades:
            return RiskDecision(
                False,
                f"max_open_trades ({open_trades}/{self.max_open_trades})",
            )

        if proposed_stake is not None:
            stake_check = self.validate_stake(proposed_stake, balance)
            if not stake_check:
                return stake_check

        return RiskDecision(True, "ok")

    def validate_stake(self, stake: float, account_balance: float) -> RiskDecision:
        """Check stake size against min/max and balance."""
        try:
            stake_f = float(stake)
            bal = float(account_balance)
        except (TypeError, ValueError):
            return RiskDecision(False, "stake_invalid")

        if stake_f < self.min_stake:
            return RiskDecision(
                False, f"stake_below_min ({stake_f:.2f} < {self.min_stake:.2f})"
            )

        if stake_f > bal:
            return RiskDecision(
                False, f"stake_exceeds_balance ({stake_f:.2f} > {bal:.2f})"
            )

        # Leave a small buffer so balance doesn't go to exact zero
        if bal - stake_f < 0:
            return RiskDecision(False, "insufficient_balance")

        cap_pct = bal * (self.max_stake_pct / 100.0) if self.max_stake_pct > 0 else bal
        hard_cap = self.max_stake if self.max_stake is not None else cap_pct
        allowed_max = min(cap_pct, hard_cap) if self.max_stake is not None else cap_pct

        if stake_f > allowed_max + 1e-9:
            return RiskDecision(
                False,
                f"stake_above_cap ({stake_f:.2f} > {allowed_max:.2f})",
            )

        return RiskDecision(True, "ok")

    def clamp_stake(self, stake: float, account_balance: float) -> float:
        """
        Reduce stake to fit risk caps (never raise it).
        Returns 0.0 if stake cannot be made valid.
        """
        try:
            stake_f = float(stake)
            bal = float(account_balance)
        except (TypeError, ValueError):
            return 0.0

        cap_pct = bal * (self.max_stake_pct / 100.0) if self.max_stake_pct > 0 else bal
        hard_cap = self.max_stake if self.max_stake is not None else cap_pct
        allowed_max = min(cap_pct, hard_cap, bal)

        adjusted = min(stake_f, allowed_max)
        if adjusted < self.min_stake:
            return 0.0
        return round(adjusted, 2)

    def record_trade_result(self, profit: float) -> None:
        self._maybe_reset_daily()
        try:
            pnl = float(profit)
        except (TypeError, ValueError):
            pnl = 0.0

        self.daily_pnl += pnl
        self.trades_today += 1

        if pnl < 0:
            self.consecutive_losses += 1
            self.losses_today += 1
            logger.info(
                "Loss recorded pnl=%.2f consecutive=%s daily_pnl=%.2f",
                pnl,
                self.consecutive_losses,
                self.daily_pnl,
            )
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.pause(reason=f"{self.consecutive_losses} consecutive losses")
        elif pnl > 0:
            self.consecutive_losses = 0
            self.wins_today += 1
            logger.info("Win recorded pnl=%.2f daily_pnl=%.2f", pnl, self.daily_pnl)
        else:
            # push — do not break streak harshly, but don't count as win
            logger.info("Push recorded pnl=0 daily_pnl=%.2f", self.daily_pnl)

    def snapshot(self) -> dict:
        return {
            "daily_pnl": self.daily_pnl,
            "daily_loss": self.daily_loss_amount(),
            "consecutive_losses": self.consecutive_losses,
            "trades_today": self.trades_today,
            "wins_today": self.wins_today,
            "losses_today": self.losses_today,
            "paused": self.is_paused(),
            "paused_until": self.paused_until.isoformat() if self.paused_until else None,
            "session_start_balance": self.session_start_balance,
            "last_known_balance": self.last_known_balance,
            "max_open_trades": self.max_open_trades,
            "min_balance": self.min_balance,
            "max_stake_pct": self.max_stake_pct,
        }
