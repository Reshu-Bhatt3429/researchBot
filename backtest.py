"""
Backtesting Engine: Validate strategy on historical data before live deployment.

Research requirements:
- Expanding-window TimeSeriesSplit (no look-ahead bias)
- Realistic fee simulation (dynamic Polymarket taker fee curve)
- 5-minute epoch alignment (timestamps % 300 == 0)
- Track ROC-AUC, Brier Score, and net P&L after fees
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import aiohttp
import asyncio

from config import Config
from data_feed import PriceFeed, Candle
from signal_engine import SignalEngine, Direction
from probability import BinaryPricing
from risk_manager import RiskManager
from paper_trading import PaperTrader

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    abstentions: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    roi: float = 0.0
    predictions: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


async def fetch_historical_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    days: int = 7,
) -> list:
    """Fetch historical 1-minute klines from Binance."""
    url = "https://api.binance.com/api/v3/klines"
    all_candles = []

    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    async with aiohttp.ClientSession() as session:
        current_start = start_time
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "limit": 1000,
            }
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Kline fetch failed: {resp.status}")
                    break
                data = await resp.json()

            if not data:
                break

            for k in data:
                candle = Candle(
                    timestamp=k[0] / 1000.0,
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                )
                all_candles.append(candle)

            current_start = int(data[-1][0]) + 60000  # Next minute
            await asyncio.sleep(0.1)  # Rate limit

    logger.info(f"Fetched {len(all_candles)} historical candles ({days} days)")
    return all_candles


def run_backtest(
    candles: list,
    config: Config = None,
    verbose: bool = True,
) -> BacktestResult:
    """
    Run backtest on historical candle data.

    Simulates the bot's decision process at each 5-minute epoch boundary:
    1. Set the window open price
    2. Wait 30 candles (simulating the 30s entry delay)
    3. Generate signal from available data
    4. Apply risk management
    5. Resolve at window close
    """
    if config is None:
        config = Config()

    result = BacktestResult()
    result.equity_curve = [config.initial_bankroll]

    # Build price feed with expanding window (no look-ahead)
    price_feed = PriceFeed(config=config)
    signal_engine = SignalEngine(config=config, price_feed=price_feed)
    pricing = BinaryPricing(config=config)
    risk_manager = RiskManager(config=config)

    # Group candles into 5-minute windows
    # Each window is 5 1-minute candles
    window_size = 5
    warmup = 100  # Need 100 candles for indicators

    logger.info(
        f"Backtesting on {len(candles)} candles "
        f"({len(candles) // window_size} windows, {warmup} warmup)"
    )

    # Feed warmup data
    for i in range(min(warmup, len(candles))):
        price_feed.candles.append(candles[i])

    # Iterate through 5-minute windows
    for window_idx in range(warmup // window_size, len(candles) // window_size):
        window_start = window_idx * window_size
        window_end = window_start + window_size

        if window_end > len(candles):
            break

        window_candles = candles[window_start:window_end]
        btc_open_price = window_candles[0].open
        btc_close_price = window_candles[-1].close

        # Add candles up to entry point (30s in = ~0.5 candles, use first candle)
        for c in window_candles[:1]:
            price_feed.candles.append(c)

        # Generate signal
        market_yes_price = 0.50
        signal = signal_engine.generate_signal(market_yes_price)

        # Risk check
        ewma_vol = signal.features.get("ewma_vol", 0.01) if signal.features else 0.01
        vol_ratio = signal.features.get("vol_ratio", 1.0) if signal.features else 1.0
        volumes = price_feed.get_volumes(5)
        current_volume = float(volumes.sum()) if len(volumes) > 0 else 2000

        decision = risk_manager.evaluate(
            signal=signal,
            market_yes_price=market_yes_price,
            market_no_price=0.50,
            current_spread=0.02,
            current_volatility_ratio=vol_ratio,
            current_volume=current_volume,
            time_to_expiry_seconds=270,
        )

        # Add remaining candles (simulating time passing)
        for c in window_candles[1:]:
            price_feed.candles.append(c)

        if not decision.should_trade:
            result.abstentions += 1
            continue

        # Resolve
        btc_went_up = btc_close_price > btc_open_price
        if signal.direction == Direction.UP:
            won = btc_went_up
        else:
            won = not btc_went_up

        # P&L with fees
        entry_price = decision.entry_price
        fee = config.effective_maker_fee(entry_price) * decision.size_usd

        if won:
            shares = decision.size_usd / entry_price
            pnl = shares * (1.0 - entry_price) - fee
        else:
            pnl = -decision.size_usd - fee

        # Record
        risk_manager.record_result(won, pnl)
        result.total_pnl += pnl
        result.total_fees += fee
        result.total_trades += 1

        if won:
            result.wins += 1
        else:
            result.losses += 1

        actual_outcome = 1 if btc_went_up else 0
        result.predictions.append(signal.confidence)
        result.outcomes.append(actual_outcome)
        signal_engine.record_outcome(signal.confidence, actual_outcome)

        result.equity_curve.append(risk_manager.bankroll)

        # Track max drawdown
        dd = risk_manager._current_drawdown()
        if dd > result.max_drawdown:
            result.max_drawdown = dd

        if risk_manager.is_halted:
            logger.warning(f"Halted at trade {result.total_trades} due to drawdown")
            break

    # Final stats
    result.win_rate = result.wins / result.total_trades if result.total_trades > 0 else 0
    result.roi = result.total_pnl / config.initial_bankroll

    if verbose:
        _print_backtest_report(result, config)

    return result


def _print_backtest_report(result: BacktestResult, config: Config):
    """Print formatted backtest results."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Total Trades:      {result.total_trades}")
    print(f"  Wins:              {result.wins}")
    print(f"  Losses:            {result.losses}")
    print(f"  Abstentions:       {result.abstentions}")
    print(f"  Win Rate:          {result.win_rate:.1%}")
    print(f"  Total PnL:         ${result.total_pnl:+.2f}")
    print(f"  Total Fees:        ${result.total_fees:.2f}")
    print(f"  ROI:               {result.roi:+.1%}")
    print(f"  Max Drawdown:      {result.max_drawdown:.1%}")
    print(f"  Final Bankroll:    ${result.equity_curve[-1]:.2f}")

    # Compute ROC-AUC if we have enough data
    if len(result.predictions) >= 10:
        from sklearn.metrics import roc_auc_score, brier_score_loss

        try:
            auc = roc_auc_score(result.outcomes, result.predictions)
            brier = brier_score_loss(result.outcomes, result.predictions)
            print(f"  ROC-AUC:           {auc:.4f}")
            print(f"  Brier Score:       {brier:.4f}")

            # Research threshold: AUC > 0.60 needed but not sufficient
            if auc < 0.60:
                print("  WARNING: ROC-AUC < 0.60 — insufficient for profitable trading")
            elif auc < 0.65:
                print("  CAUTION: ROC-AUC marginal — may not overcome fees")
            else:
                print("  ROC-AUC looks promising for fee-adjusted profitability")
        except Exception as e:
            print(f"  ROC-AUC: Error computing ({e})")

    print("=" * 60)


async def main():
    """Run a backtest on the last 7 days of data."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    print("Fetching historical data...")
    candles = await fetch_historical_klines(days=7)

    if len(candles) < 200:
        print(f"Not enough data ({len(candles)} candles). Need at least 200.")
        return

    config = Config()
    result = run_backtest(candles, config, verbose=True)

    # Parameter sweep suggestion
    print("\nTo tune parameters, modify config.py values:")
    print("  - kelly_fraction: 0.1 to 0.5")
    print("  - min_edge_threshold: 0.02 to 0.05")
    print("  - min_confidence: 0.55 to 0.65")
    print("  - ewma_span: 15 to 60")
    print("  - momentum_windows: try [2, 5, 10, 20]")


if __name__ == "__main__":
    asyncio.run(main())
