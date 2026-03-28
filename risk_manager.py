"""
Risk Management: Fractional Kelly Criterion + 7-Condition Abstention.

Research alpha:
- Fractional Kelly (Quarter-Kelly) to survive heavy-tail distributions
- 7-condition circuit breaker preventing trades without statistical edge
- Platt calibration monitoring via Brier Score, Log Loss, Murphy decomposition
- Max drawdown halt, consecutive loss tracking
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import Config
from signal_engine import Signal, Direction

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    should_trade: bool
    direction: Direction
    size_usd: float
    entry_price: float  # Limit order price
    confidence: float
    edge: float
    abstention_reason: Optional[str] = None


@dataclass
class RiskManager:
    """
    Implements Fractional Kelly Criterion and 7-condition abstention rule.

    From research (Joicodev framework):
    - Standard Kelly is catastrophic for heavy-tail BTC distributions
    - Quarter-Kelly preserves capital during drawdowns
    - 7-condition abstention prevents trading without measurable edge
    """
    config: Config
    bankroll: float = 0.0
    peak_bankroll: float = 0.0
    consecutive_losses: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    _halted: bool = False

    def __post_init__(self):
        if self.bankroll == 0:
            self.bankroll = self.config.initial_bankroll
        self.peak_bankroll = self.bankroll

    def evaluate(
        self,
        signal: Signal,
        market_yes_price: float,
        market_no_price: float,
        current_spread: float,
        current_volatility_ratio: float,
        current_volume: float,
        time_to_expiry_seconds: float,
    ) -> TradeDecision:
        """
        Run the 7-condition abstention check, then compute Kelly sizing.

        The 7 abstention conditions (from research):
        1. No measurable edge over market price
        2. Volatility too extreme (> 3x normal)
        3. Insufficient volume
        4. Spread too wide
        5. Too many consecutive losses
        6. Too close to expiry
        7. Drawdown exceeds maximum threshold
        """
        # === CONDITION 1: Minimum edge threshold ===
        min_fee = self.config.taker_fee_at_price(market_yes_price)
        required_edge = max(self.config.min_edge_threshold, min_fee)

        if signal.edge < required_edge:
            return self._abstain(
                signal,
                f"Edge {signal.edge:.4f} < required {required_edge:.4f} "
                f"(fee={min_fee:.4f})",
            )

        # === CONDITION 2: Extreme volatility ===
        if current_volatility_ratio > self.config.max_volatility_ratio:
            return self._abstain(
                signal,
                f"Volatility ratio {current_volatility_ratio:.2f} > "
                f"max {self.config.max_volatility_ratio}",
            )

        # === CONDITION 3: Insufficient volume ===
        if current_volume < self.config.min_volume_threshold:
            return self._abstain(
                signal,
                f"Volume {current_volume:.0f} < min {self.config.min_volume_threshold}",
            )

        # === CONDITION 4: Spread too wide ===
        if current_spread > self.config.max_spread_pct:
            return self._abstain(
                signal,
                f"Spread {current_spread:.4f} > max {self.config.max_spread_pct}",
            )

        # === CONDITION 5: Consecutive losses ===
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return self._abstain(
                signal,
                f"Consecutive losses {self.consecutive_losses} >= "
                f"max {self.config.max_consecutive_losses}",
            )

        # === CONDITION 6: Too close to expiry ===
        if time_to_expiry_seconds < self.config.min_time_to_expiry_seconds:
            return self._abstain(
                signal,
                f"Time to expiry {time_to_expiry_seconds:.0f}s < "
                f"min {self.config.min_time_to_expiry_seconds}s",
            )

        # === CONDITION 7: Maximum drawdown ===
        current_drawdown = self._current_drawdown()
        if current_drawdown > self.config.max_drawdown_pct:
            self._halted = True
            return self._abstain(
                signal,
                f"Drawdown {current_drawdown:.2%} > max {self.config.max_drawdown_pct:.2%}. "
                f"HALTED.",
            )

        # === CONFIDENCE CHECK ===
        if signal.confidence < self.config.min_confidence:
            return self._abstain(
                signal,
                f"Confidence {signal.confidence:.4f} < min {self.config.min_confidence}",
            )

        # === All conditions passed: compute Kelly sizing ===
        size_usd = self._kelly_size(signal, market_yes_price)

        # Determine entry price based on maker/taker preference
        if self.config.use_maker_orders:
            entry_price = self._compute_maker_price(
                signal.direction, market_yes_price, market_no_price
            )
        else:
            entry_price = (
                market_yes_price
                if signal.direction == Direction.UP
                else market_no_price
            )

        return TradeDecision(
            should_trade=True,
            direction=signal.direction,
            size_usd=size_usd,
            entry_price=entry_price,
            confidence=signal.confidence,
            edge=signal.edge,
        )

    def _kelly_size(self, signal: Signal, market_price: float) -> float:
        """
        Fractional Kelly Criterion position sizing.

        Kelly fraction f* = (p*b - q) / b
        where:
            p = probability of winning (our estimated confidence)
            q = 1 - p
            b = odds (payout / risk)

        For binary options at price p_market:
            b = (1 - p_market) / p_market  (win $1-p for risking p)

        Then apply fractional multiplier (Quarter-Kelly by default).
        """
        p = signal.confidence
        q = 1.0 - p

        # Account for fees in the odds calculation
        fee = self.config.taker_fee_at_price(market_price)
        effective_payout = 1.0 - market_price - fee  # Net win after fees
        risk = market_price  # Amount risked

        if risk <= 0 or effective_payout <= 0:
            return 0.0

        b = effective_payout / risk  # Net odds

        # Kelly fraction
        kelly_f = (p * b - q) / b
        kelly_f = max(kelly_f, 0.0)  # Never negative

        # Apply fractional Kelly multiplier
        position_fraction = kelly_f * self.config.kelly_fraction

        # Cap at maximum position size
        position_fraction = min(position_fraction, self.config.max_position_pct)

        size_usd = self.bankroll * position_fraction

        # Round to tick size
        size_usd = math.floor(size_usd / self.config.tick_size) * self.config.tick_size

        return max(size_usd, 0.0)

    def _compute_maker_price(
        self,
        direction: Direction,
        market_yes_price: float,
        market_no_price: float,
    ) -> float:
        """
        Compute maker limit order price to capture rebate.

        Research: maker orders earn 20% rebate, critical for overcoming fees.
        Place order 1 tick inside the best bid/ask.
        """
        tick = self.config.tick_size
        offset = self.config.maker_offset_ticks * tick

        if direction == Direction.UP:
            # Buy YES: place bid slightly above current best bid
            # Best bid is approximately market_yes_price - spread/2
            price = market_yes_price - offset
            price = max(price, tick)  # Minimum $0.01
        else:
            # Buy NO: place bid slightly above current best bid for NO
            price = market_no_price - offset
            price = max(price, tick)

        # Round to tick
        price = round(price / tick) * tick
        return price

    def record_result(self, won: bool, pnl: float):
        """Update bankroll and tracking after a trade resolves."""
        self.bankroll += pnl
        self.total_trades += 1

        if won:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1

        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll

        win_rate = self.wins / self.total_trades if self.total_trades > 0 else 0
        logger.info(
            f"Trade result: {'WIN' if won else 'LOSS'} | PnL: ${pnl:+.2f} | "
            f"Bankroll: ${self.bankroll:.2f} | "
            f"Win rate: {win_rate:.1%} ({self.wins}/{self.total_trades}) | "
            f"Drawdown: {self._current_drawdown():.1%}"
        )

    def _current_drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_bankroll <= 0:
            return 0.0
        return (self.peak_bankroll - self.bankroll) / self.peak_bankroll

    def _abstain(self, signal: Signal, reason: str) -> TradeDecision:
        logger.debug(f"ABSTAIN: {reason}")
        return TradeDecision(
            should_trade=False,
            direction=signal.direction,
            size_usd=0.0,
            entry_price=0.0,
            confidence=signal.confidence,
            edge=signal.edge,
            abstention_reason=reason,
        )

    @property
    def is_halted(self) -> bool:
        return self._halted

    def reset_halt(self):
        """Manual reset after drawdown halt."""
        self._halted = False
        logger.info("Drawdown halt reset manually")

    def get_stats(self) -> dict:
        return {
            "bankroll": self.bankroll,
            "peak_bankroll": self.peak_bankroll,
            "drawdown": self._current_drawdown(),
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.wins / self.total_trades if self.total_trades > 0 else 0,
            "consecutive_losses": self.consecutive_losses,
            "halted": self._halted,
        }

    def compute_model_health(
        self, predictions: list, outcomes: list
    ) -> dict:
        """
        Compute model health metrics (research: Brier Score, Log Loss,
        Murphy decomposition).
        """
        if not predictions or not outcomes:
            return {"brier_score": None, "log_loss": None}

        preds = np.array(predictions)
        actual = np.array(outcomes)

        # Brier Score: mean squared error of probabilities
        brier = float(np.mean((preds - actual) ** 2))

        # Log Loss
        preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
        log_loss = -float(
            np.mean(
                actual * np.log(preds_clipped)
                + (1 - actual) * np.log(1 - preds_clipped)
            )
        )

        # Murphy decomposition: reliability + resolution - uncertainty
        # Simplified version
        n = len(actual)
        base_rate = np.mean(actual)
        uncertainty = base_rate * (1 - base_rate)

        return {
            "brier_score": brier,
            "log_loss": log_loss,
            "base_rate": float(base_rate),
            "uncertainty": float(uncertainty),
            "n_samples": n,
        }
