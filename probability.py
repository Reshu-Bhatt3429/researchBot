"""
Black-Scholes Binary Option Pricing with EWMA Volatility.

Research alpha:
- Adapts Black-Scholes-Merton for cash-or-nothing binary options
- Uses EWMA volatility instead of static historical vol
- Logit-space momentum adjustments to keep probabilities in [0, 1]
- Computes theoretical fair value to detect mispricing vs orderbook
"""

import math
import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from scipy.special import expit, logit

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class BinaryPricing:
    """
    Prices 5-minute Bitcoin binary options using adapted Black-Scholes.

    A cash-or-nothing binary call pays $1 if S_T > K, $0 otherwise.
    P(S_T > K) = N(d2) where d2 = (ln(S/K) + (r - σ²/2)T) / (σ√T)

    For 5-minute Polymarket markets:
    - S = current BTC price
    - K = opening price of the 5-minute window
    - T = time remaining in seconds / seconds_per_year
    - σ = EWMA volatility (annualized)
    - r = risk-free rate
    """
    config: Config

    # Constants
    SECONDS_PER_YEAR: float = 365.25 * 24 * 3600

    def binary_call_probability(
        self,
        spot: float,
        strike: float,
        time_remaining_seconds: float,
        ewma_vol_1m: float,
        momentum_adjustment: float = 0.0,
    ) -> float:
        """
        Compute P(BTC > strike at expiry) using Black-Scholes binary pricing.

        Args:
            spot: Current BTC price
            strike: Opening price of 5-minute window
            time_remaining_seconds: Seconds until binary resolution
            ewma_vol_1m: EWMA volatility computed on 1-minute returns
            momentum_adjustment: Logit-space momentum shift

        Returns:
            Probability in [0, 1] that BTC closes above strike
        """
        if time_remaining_seconds <= 0:
            # Already expired: deterministic
            return 1.0 if spot > strike else 0.0

        if strike <= 0 or spot <= 0:
            return 0.5

        # Annualize the 1-minute EWMA volatility
        # σ_annual = σ_1m * √(minutes_per_year)
        minutes_per_year = 365.25 * 24 * 60
        sigma_annual = ewma_vol_1m * math.sqrt(minutes_per_year)

        # Clamp volatility to reasonable bounds
        sigma_annual = max(sigma_annual, 0.01)  # Minimum 1% annual vol
        sigma_annual = min(sigma_annual, 5.0)   # Maximum 500% annual vol

        # Time to expiry in years
        T = time_remaining_seconds / self.SECONDS_PER_YEAR

        r = self.config.risk_free_rate

        # Black-Scholes d2 for binary option
        sqrt_T = math.sqrt(T)
        d2 = (math.log(spot / strike) + (r - 0.5 * sigma_annual**2) * T) / (
            sigma_annual * sqrt_T
        )

        # Base probability from normal CDF
        prob = norm.cdf(d2)

        # Apply momentum adjustment in logit space (research: prevents overflow)
        if abs(momentum_adjustment) > 1e-6:
            prob = self._logit_space_adjust(prob, momentum_adjustment)

        return float(np.clip(prob, 0.001, 0.999))

    def binary_put_probability(
        self,
        spot: float,
        strike: float,
        time_remaining_seconds: float,
        ewma_vol_1m: float,
        momentum_adjustment: float = 0.0,
    ) -> float:
        """P(BTC < strike at expiry) = 1 - P(BTC > strike)."""
        return 1.0 - self.binary_call_probability(
            spot, strike, time_remaining_seconds, ewma_vol_1m, momentum_adjustment
        )

    def compute_edge(
        self,
        theoretical_prob: float,
        market_price: float,
        is_yes: bool = True,
    ) -> float:
        """
        Compute expected value edge: theoretical probability vs market price.

        For YES token at price p:
            EV = theoretical_prob * (1 - p) - (1 - theoretical_prob) * p
               = theoretical_prob - p

        Edge must exceed taker fee to be profitable.
        """
        if is_yes:
            edge = theoretical_prob - market_price
        else:
            edge = (1.0 - theoretical_prob) - market_price
        return edge

    def min_profitable_edge(self, market_price: float) -> float:
        """
        Minimum edge needed to overcome Polymarket taker fee.
        Uses the dynamic fee curve from config.
        """
        fee = self.config.taker_fee_at_price(market_price)
        return fee

    def _logit_space_adjust(self, prob: float, adjustment: float) -> float:
        """
        Apply adjustment in logit space to keep result in (0, 1).

        Research: "Raw momentum signals are calculated and injected into
        probability models strictly within logit space to ensure extreme
        momentum data does not push final probabilities outside [0, 1]."
        """
        # Clamp to prevent logit of 0 or 1
        prob = np.clip(prob, 0.001, 0.999)
        log_odds = logit(prob)
        adjusted_log_odds = log_odds + adjustment
        # Clamp the adjusted logit to prevent extreme probabilities
        adjusted_log_odds = np.clip(adjusted_log_odds, -6.0, 6.0)
        return float(expit(adjusted_log_odds))

    def compute_fair_value(
        self,
        spot: float,
        strike: float,
        time_remaining_seconds: float,
        ewma_vol_1m: float,
        momentum_features: dict,
    ) -> dict:
        """
        Compute full fair value analysis for both sides of the binary.

        Returns dict with theoretical prices, edges, and trade recommendation.
        """
        # Compute momentum adjustment from features
        mom_adj = self._momentum_to_logit_adjustment(momentum_features)

        prob_up = self.binary_call_probability(
            spot, strike, time_remaining_seconds, ewma_vol_1m, mom_adj
        )
        prob_down = 1.0 - prob_up

        return {
            "prob_up": prob_up,
            "prob_down": prob_down,
            "momentum_adjustment": mom_adj,
            "sigma_annual": ewma_vol_1m * math.sqrt(365.25 * 24 * 60),
            "time_remaining": time_remaining_seconds,
            "spot": spot,
            "strike": strike,
        }

    def _momentum_to_logit_adjustment(self, features: dict) -> float:
        """
        Convert momentum features to a logit-space adjustment.
        Moderate scaling to avoid overreaction to noise.
        """
        if not features:
            return 0.0

        # Weighted combination of momentum signals
        mom_3 = features.get("momentum_3", 0.0)
        mom_6 = features.get("momentum_6", 0.0)
        trend_10 = features.get("trend_10m", 0.0)

        # Scale: 1% momentum -> ~0.1 logit adjustment
        adjustment = 10.0 * (0.5 * mom_3 + 0.3 * mom_6 + 0.2 * trend_10)

        # Clamp to prevent extreme adjustments
        return float(np.clip(adjustment, -1.0, 1.0))
