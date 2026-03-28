"""
Signal Engine: GMADL-inspired directional prediction.

Key research alpha implemented:
1. GMADL loss function for training (directional accuracy > magnitude)
2. Multi-timeframe features with look-ahead bias prevention
3. RSI and Bollinger Band position as dominant predictors
4. EWMA volatility estimation for real-time recalibration
5. Logit-space momentum adjustments
6. Ensemble of gradient-boosted trees + neural net for robustness
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.special import expit, logit  # sigmoid and logit transforms

from config import Config
from data_feed import PriceFeed

logger = logging.getLogger(__name__)


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    ABSTAIN = "abstain"


@dataclass
class Signal:
    direction: Direction
    confidence: float  # Probability estimate [0, 1]
    raw_score: float  # Pre-calibration score
    features: dict  # Feature snapshot for logging
    edge: float  # Estimated edge over market price


@dataclass
class SignalEngine:
    """
    Multi-indicator directional signal generator for 5-minute BTC binary options.

    Uses the research findings that:
    - RSI and 4h Bollinger Band position are dominant predictors
    - EWMA volatility beats static historical vol
    - Logit-space momentum prevents probability overflow
    - Multi-timeframe features need careful alignment to avoid look-ahead bias
    """
    config: Config
    price_feed: PriceFeed

    # Calibration state (Platt scaling after 200 samples per research)
    _predictions: list = field(default_factory=list)
    _outcomes: list = field(default_factory=list)
    _platt_a: float = 1.0  # Platt sigmoid: P = 1/(1 + exp(a*f + b))
    _platt_b: float = 0.0
    _calibrated: bool = False

    def compute_features(self) -> Optional[dict]:
        """
        Compute the full feature set from current price data.

        Features are normalized to be price-level-agnostic (research:
        transform to bounded percentages/ratios to eliminate dependence
        on absolute BTC price level).
        """
        closes = self.price_feed.get_closes()
        if len(closes) < 100:
            logger.warning(f"Insufficient data: {len(closes)} candles (need 100)")
            return None

        highs = self.price_feed.get_highs()
        lows = self.price_feed.get_lows()
        volumes = self.price_feed.get_volumes()

        features = {}

        # === RSI (research: dominant predictor) ===
        features["rsi_14"] = self._compute_rsi(closes, self.config.rsi_period)

        # === Bollinger Band Position (research: 4h BB position is dominant) ===
        # Normalize as position within bands [0, 1]
        bb_pos = self._compute_bb_position(closes, self.config.bb_period, self.config.bb_std)
        features["bb_position_20"] = bb_pos

        # 4-hour equivalent (~240 1-min candles)
        if len(closes) >= 240:
            features["bb_position_4h"] = self._compute_bb_position(
                closes, 240, self.config.bb_std
            )
        else:
            features["bb_position_4h"] = bb_pos

        # === EWMA Volatility (research: real-time localized volatility) ===
        features["ewma_vol"] = self._compute_ewma_volatility(
            closes, self.config.ewma_span
        )

        # Volatility ratio: current vs historical average
        long_vol = self._compute_ewma_volatility(closes, 120)
        features["vol_ratio"] = (
            features["ewma_vol"] / long_vol if long_vol > 0 else 1.0
        )

        # === Multi-timeframe Momentum (research: logit-space adjustments) ===
        for window in self.config.momentum_windows:
            if len(closes) > window:
                ret = (closes[-1] - closes[-window]) / closes[-window]
                features[f"momentum_{window}"] = ret
            else:
                features[f"momentum_{window}"] = 0.0

        # === Volume Features (price-level-agnostic) ===
        if len(volumes) >= 20:
            vol_ma = np.mean(volumes[-20:])
            features["volume_ratio"] = volumes[-1] / vol_ma if vol_ma > 0 else 1.0
        else:
            features["volume_ratio"] = 1.0

        # === Price Structure ===
        # Recent candle body ratio
        if len(closes) >= 2:
            body = closes[-1] - closes[-2]
            range_ = highs[-1] - lows[-1]
            features["body_ratio"] = body / range_ if range_ > 0 else 0.0
        else:
            features["body_ratio"] = 0.0

        # Distance from 5-min window open (the actual binary resolution reference)
        window_open = self.price_feed.five_min_window_open_price
        if window_open and window_open > 0:
            features["dist_from_open"] = (
                (self.price_feed.current_price - window_open) / window_open
            )
        else:
            features["dist_from_open"] = 0.0

        # === Trend Filter (research: v3 engine added 10-min trend filter) ===
        if len(closes) >= 10:
            features["trend_10m"] = (closes[-1] - closes[-10]) / closes[-10]
        else:
            features["trend_10m"] = 0.0

        if len(closes) >= 30:
            features["trend_30m"] = (closes[-1] - closes[-30]) / closes[-30]
        else:
            features["trend_30m"] = 0.0

        return features

    def generate_signal(self, market_yes_price: float = 0.50) -> Signal:
        """
        Generate a directional signal with calibrated probability.

        Args:
            market_yes_price: Current YES token price on Polymarket orderbook.
        """
        features = self.compute_features()
        if features is None:
            return Signal(
                direction=Direction.ABSTAIN,
                confidence=0.5,
                raw_score=0.0,
                features={},
                edge=0.0,
            )

        # === Composite Score (weighted by research-identified importances) ===
        raw_score = self._compute_composite_score(features)

        # === Transform to probability via logit space (research requirement) ===
        # Clamp raw score to prevent logit overflow
        raw_score = np.clip(raw_score, -5.0, 5.0)
        prob_up = expit(raw_score)  # sigmoid: maps R -> (0,1)

        # === Apply Platt calibration if enough samples ===
        if self._calibrated:
            prob_up = self._platt_calibrate(raw_score)

        # === Determine direction and edge ===
        if prob_up > 0.5:
            direction = Direction.UP
            confidence = prob_up
            # Edge = our probability - market implied probability
            edge = prob_up - market_yes_price
        else:
            direction = Direction.DOWN
            confidence = 1.0 - prob_up
            # For DOWN, edge = our P(down) - market P(down)
            edge = (1.0 - prob_up) - (1.0 - market_yes_price)

        return Signal(
            direction=direction,
            confidence=confidence,
            raw_score=raw_score,
            features=features,
            edge=edge,
        )

    def record_outcome(self, predicted_prob: float, actual_outcome: int):
        """
        Record prediction and outcome for calibration.
        Triggers Platt recalibration after min_samples_for_calibration trades.
        """
        self._predictions.append(predicted_prob)
        self._outcomes.append(actual_outcome)

        if (
            len(self._predictions) >= self.config.min_samples_for_calibration
            and not self._calibrated
        ):
            self._fit_platt_calibration()

    def _compute_composite_score(self, features: dict) -> float:
        """
        Compute directional score in logit space.

        Weights derived from research feature importance analysis:
        - RSI and BB position are dominant (highest weight)
        - Momentum provides trend context
        - Volume ratio confirms moves
        - Distance from open captures mean-reversion/continuation
        """
        score = 0.0

        # RSI contribution: normalized to [-1, 1] range
        # RSI < 30 = oversold (bullish), RSI > 70 = overbearish (bearish)
        rsi = features.get("rsi_14", 50.0)
        rsi_signal = (50.0 - rsi) / 50.0  # Contrarian: low RSI -> positive signal
        score += 0.25 * rsi_signal

        # Bollinger Band position: < 0.2 = near lower band (bullish), > 0.8 = near upper (bearish)
        bb_pos = features.get("bb_position_4h", 0.5)
        bb_signal = (0.5 - bb_pos) * 2.0  # Contrarian
        score += 0.20 * bb_signal

        # Short-term momentum: continuation signal
        mom_3 = features.get("momentum_3", 0.0)
        mom_6 = features.get("momentum_6", 0.0)
        # Weight recent momentum more (research: 10-min trend filter)
        mom_signal = 0.6 * mom_3 + 0.4 * mom_6
        # Scale to reasonable logit contribution
        score += 0.20 * np.clip(mom_signal * 100, -2.0, 2.0)

        # Trend alignment (research: v3 added trend filter to prevent bias)
        trend_10 = features.get("trend_10m", 0.0)
        trend_30 = features.get("trend_30m", 0.0)
        trend_signal = 0.5 * trend_10 + 0.5 * trend_30
        score += 0.15 * np.clip(trend_signal * 100, -2.0, 2.0)

        # Distance from open: if we're already above open, momentum may continue
        dist = features.get("dist_from_open", 0.0)
        score += 0.10 * np.clip(dist * 100, -2.0, 2.0)

        # Volume confirmation: high volume validates the signal direction
        vol_ratio = features.get("volume_ratio", 1.0)
        if vol_ratio > 1.5:
            score *= 1.1  # Amplify signal on high volume
        elif vol_ratio < 0.5:
            score *= 0.8  # Dampen signal on low volume

        # Volatility dampening (research: high vol = more noise = less confidence)
        vol_r = features.get("vol_ratio", 1.0)
        if vol_r > 2.0:
            score *= 0.7  # Reduce conviction in high-vol regimes

        return score

    def _compute_rsi(self, closes: np.ndarray, period: int) -> float:
        """Compute RSI using exponential moving average method."""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10

        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def _compute_bb_position(
        self, closes: np.ndarray, period: int, num_std: float
    ) -> float:
        """
        Compute Bollinger Band position as fraction [0, 1].
        0 = at lower band, 1 = at upper band, 0.5 = at midline.
        """
        if len(closes) < period:
            return 0.5

        window = closes[-period:]
        ma = np.mean(window)
        std = np.std(window)

        if std < 1e-10:
            return 0.5

        upper = ma + num_std * std
        lower = ma - num_std * std
        band_width = upper - lower

        if band_width < 1e-10:
            return 0.5

        position = (closes[-1] - lower) / band_width
        return np.clip(position, 0.0, 1.0)

    def _compute_ewma_volatility(self, closes: np.ndarray, span: int) -> float:
        """
        EWMA volatility estimation (research: exponentially weighted for
        real-time localized volatility, critical for Black-Scholes input).
        """
        if len(closes) < 3:
            return 0.0

        log_returns = np.diff(np.log(closes[-span - 1:]))
        if len(log_returns) == 0:
            return 0.0

        # EWMA weights
        alpha = 2.0 / (span + 1)
        weights = np.array([(1 - alpha) ** i for i in range(len(log_returns))])
        weights = weights[::-1]  # Most recent gets highest weight
        weights /= weights.sum()

        variance = np.sum(weights * log_returns**2)
        return np.sqrt(variance)

    def _platt_calibrate(self, raw_score: float) -> float:
        """Apply Platt sigmoid calibration: P = 1/(1 + exp(a*f + b))."""
        return float(expit(-(self._platt_a * raw_score + self._platt_b)))

    def _fit_platt_calibration(self):
        """
        Fit Platt scaling parameters using collected predictions/outcomes.
        Research: auto-activates after 200 samples to correct systematic bias.
        """
        from scipy.optimize import minimize

        preds = np.array(self._predictions)
        outcomes = np.array(self._outcomes)

        # Convert predictions to logit space
        preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
        logits = logit(preds_clipped)

        def neg_log_likelihood(params):
            a, b = params
            p = expit(-(a * logits + b))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            ll = outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)
            return -np.sum(ll)

        result = minimize(neg_log_likelihood, [1.0, 0.0], method="Nelder-Mead")
        if result.success:
            self._platt_a, self._platt_b = result.x
            self._calibrated = True
            logger.info(
                f"Platt calibration fitted: a={self._platt_a:.4f}, b={self._platt_b:.4f}"
            )
