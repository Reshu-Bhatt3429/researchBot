# Overall Decision — No Strategy Clears 12 bps Robustly

## What worked

Future movement magnitude is meaningfully predictable from one-minute BTC information. In the untouched final block, the expansion forecast's Spearman correlation with absolute future return was 0.429 at 30 minutes, 0.402 at 60, 0.383 at 90 and 0.375 at 120. The top forecast decile produced 37.78 bps average absolute 30-minute movement versus 17.84 bps unconditionally.

This validates the first half of the architecture: identify when the market is likely to move.

## What failed

The second half—direction and extraction—did not clear standardized costs:

- Direct nonlinear direction model: negative after 12 bps.
- Momentum at 15/30/60 minutes: negative.
- Rolling breakout location: negative.
- Bar-level taker imbalance: negative.
- Wick rejection and impulse reversal: negative or unstable.
- Failed-breakout reversal: unstable and too infrequent.
- Target/stop exits: worse than the simple fixed-horizon breakout.

## Near-miss candidate

The most economically interesting rule was:

1. Forecast 30-minute absolute return using only completed one-minute bars.
2. Act during the top 5% of development-calibrated expansion forecasts.
3. Place causal two-sided breakout triggers 5 bps above and below the signal close.
4. Enter whichever side triggers first; discard a bar if both levels trigger because the path is unknowable.
5. Exit 30 minutes after the original signal and prohibit overlapping trades.

| Block | Trades/day | Gross bps/trade | Net at 12 bps | PF at 12 bps | Daily Sharpe at 12 bps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 3.95 | 7.85 | -4.15 | 0.87 | -1.86 |
| Confirmation | 2.57 | 13.31 | +1.31 | 1.06 | 0.94 |
| Final | 1.71 | 11.78 | -0.22 | 0.99 | -0.12 |

The final gross result was statistically positive under the diagnostic HAC calculation, but costs consumed the entire effect. Development was already negative after 12 bps, so this is not a stable plateau and must not be described as a profitable strategy.

## Decision

Evidence label: **fragile / rejected at 12 bps**.

The research supports an expansion detector, not an executable directional strategy. The next test should use data that can reveal direction after expansion without paying away the edge: L2 imbalance, signed trade-flow persistence, liquidation prints, cross-venue leadership and actual bid/ask execution. A lower-cost maker implementation is worth measuring, but cost assumptions must be verified rather than chosen to make the backtest positive.

The target/stop family was added after the fixed-horizon final results were inspected, so its final-period results are exploratory rather than untouched out-of-sample evidence.
