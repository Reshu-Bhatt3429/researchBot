# BTC Expansion Strategy Research

Reproducible research into a narrow question: **can one-minute BTC information predict rare 30–120 minute expansion, and can a causal direction or breakout rule convert that prediction into profit after costs?**

## Result

Expansion magnitude was forecastable, direction was not reliably monetizable.

- Final-block Spearman correlation between predicted and realized absolute movement: **0.38–0.43** across 30–120 minute horizons.
- At 30 minutes, predicted top-decile events averaged **37.78 bps** of absolute close movement versus **17.84 bps** unconditionally.
- The best practical rule was an expansion-gated, two-sided 5 bps breakout held for 30 minutes.
- It generated about **1.7 final-period trades/day**, **+11.78 bps gross/trade**, but **−0.22 bps/trade after a 12 bps round trip**.
- Its final gross daily Sharpe was approximately **5.60**; after 12 bps it fell to **−0.12**.
- Fixed directional models, momentum, breakout location, taker imbalance, wick reversal, failed-breakout reversal, and target/stop exits did not survive 12 bps.

Evidence label: **fragile / not ready for paper trading**. The near-break-even breakout may be viable only if verified all-in round-trip cost is materially below 11.78 bps and it survives new data, another venue, and full perturbation tests.

## Research architecture

1. Forecast future absolute return from lagged realized volatility, range, volatility ratios, compression, volume, trade count, taker imbalance and time-of-day.
2. Separately test direction via nonlinear regression, momentum, rolling breakouts, taker flow, wick rejection and impulse reversal.
3. Test direction-free entry using two-sided stop breakouts after a high expansion forecast.
4. Test fixed-horizon and target/stop exits.
5. Use chronological 55/25/20 development, confirmation and final blocks with holding-period purges.
6. Enter only after the signal bar closes, reject ambiguous same-bar dual triggers, prohibit overlapping trades, and charge 0/12/30 bps.

## Academic motivation

- Realized-volatility persistence and heterogeneous horizons motivate HAR-style lagged volatility features. Bergsli et al. report that realized-variance HAR models outperform daily-data GARCH models for Bitcoin: [Forecasting volatility of Bitcoin](https://doi.org/10.1016/j.ribaf.2021.101540).
- Order flow, trade dynamics and limit-book information can improve short-term Bitcoin volatility forecasts: [Lensky & Hao, 2024](https://arxiv.org/abs/2304.02472).
- Cryptocurrency volatility and volume exhibit hour-of-day and within-hour periodicity across venues: [Hansen, Kim & Kimbrough](https://arxiv.org/abs/2109.12142).
- Jumps and structural breaks matter for Bitcoin volatility forecasting: [Shen, Urquhart & Wang](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3449756).

These papers support volatility forecasting, not directional profitability. That distinction is the main empirical lesson of this repository.

## Run

The input is a Binance USD-M BTCUSDT one-minute kline CSV (optionally gzip-compressed) with the standard fields used in `research.py`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest -v
python research.py --input data/BTCUSDT_1m.csv.gz --output results
python breakout_research.py --input data/BTCUSDT_1m.csv.gz --output results
python barrier_research.py --input data/BTCUSDT_1m.csv.gz --output results
```

The original experiment used 525,601 consecutive bars from 2025-08-30 16:45 UTC through 2026-08-30 16:45 UTC. Raw market data is intentionally excluded from Git; its fingerprint and lineage are in `results/data_manifest.json`.

## Files

- `research.py`: expansion forecast, directional families, chronological selection and inference.
- `breakout_research.py`: causal two-sided breakout and failed-breakout reversal tests.
- `barrier_research.py`: target/stop exit sweep.
- `results/report.md`: primary model report.
- `results/expansion_forecast.csv`: out-of-sample volatility-ranking evidence.
- `results/breakout_selected_metrics.csv`: near-break-even breakout metrics by split and cost.
- `results/*surface.csv`: complete retained parameter searches, including failures.

## Warning

This is research software, not investment advice. A backtest does not establish future profitability. The 12 bps assumption is a standardized stress case; replace it with your actual fee, spread, slippage, funding and latency estimates before any forward test.
