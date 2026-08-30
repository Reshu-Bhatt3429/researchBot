# BTC Expansion Strategy Discovery

## Decision

No selected strategy remained positive in the untouched final block after 12 bps; the tested candle/flow families are rejected for trading.

## Expansion prediction

Volatility magnitude is forecastable when rank correlation and top-decile realized movement exceed the unconditional baseline. Profitability still requires correct direction and executable selection.

## Selected candidates

| family | horizon | coverage | split | cost_bps | trade_count | trades_per_day | mean_net_bps | median_net_bps | win_rate | profit_factor | daily_sharpe | max_drawdown | mean_daily_return | standard_error | t_stat | p_value | p_value_two_sided | effective_sample_size | average_holding_minutes | cost_break_even_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direction_ml | 60 | 0.05 | development | 12 | 463 | 2.32663 | 49.5579 | 44.5475 | 0.736501 | 4.58545 | 7.75312 | -0.0352431 | 0.0115303 | 0.00352773 | 3.26848 | 0.000540644 | 0.00108129 | 199 | 60 | 61.5579 |
| direction_ml | 60 | 0.05 | confirmation | 12 | 145 | 1.59341 | -0.629924 | -6.69234 | 0.489655 | 0.981202 | -0.250143 | -0.0793786 | -0.000100372 | 0.000582494 | -0.172315 | 0.568405 | 0.86319 | 91 | 60 | 11.3701 |
| direction_ml | 60 | 0.05 | final | 12 | 83 | 1.16901 | -26.859 | -26.7869 | 0.349398 | 0.445734 | -5.38118 | -0.219725 | -0.00313986 | 0.00130707 | -2.40221 | 0.991852 | 0.0162964 | 71 | 60 | -14.859 |
| wick_rejection | 60 | 0.05 | development | 12 | 463 | 2.32663 | -9.02825 | -4.56382 | 0.485961 | 0.804137 | -2.53661 | -0.408078 | -0.00210054 | 0.000834993 | -2.51564 | 0.994059 | 0.0118816 | 199 | 60 | 2.97175 |
| wick_rejection | 60 | 0.05 | confirmation | 12 | 145 | 1.59341 | -2.83639 | -0.81971 | 0.496552 | 0.917641 | -1.15202 | -0.110404 | -0.000451952 | 0.000737338 | -0.612951 | 0.730046 | 0.539909 | 91 | 60 | 9.16361 |
| wick_rejection | 60 | 0.05 | final | 12 | 83 | 1.16901 | -27.2575 | -27.4857 | 0.313253 | 0.44601 | -6.17633 | -0.222581 | -0.00318644 | 0.00125457 | -2.53986 | 0.994455 | 0.0110896 | 71 | 60 | -15.2575 |
| impulse_reversal_15 | 120 | 0.2 | development | 12 | 805 | 4.025 | -8.89867 | -3.36771 | 0.489441 | 0.803165 | -3.30686 | -0.586028 | -0.00358171 | 0.00156627 | -2.28678 | 0.988896 | 0.0222089 | 200 | 120 | 3.10133 |
| impulse_reversal_15 | 120 | 0.2 | confirmation | 12 | 293 | 3.21978 | 3.50755 | -4.71238 | 0.464164 | 1.11118 | 1.15219 | -0.12346 | 0.00112936 | 0.00225626 | 0.500543 | 0.308346 | 0.616693 | 91 | 120 | 15.5076 |
| impulse_reversal_15 | 120 | 0.2 | final | 12 | 187 | 2.6338 | -7.97658 | -4.05088 | 0.465241 | 0.790811 | -2.23772 | -0.173245 | -0.00210087 | 0.00158377 | -1.3265 | 0.907664 | 0.184673 | 71 | 120 | 4.02342 |

## Limitations

One instrument and one year; bar-level flow only; no L2 book, liquidation feed, funding, or cross-venue validation. The final block was opened once for the three frozen confirmation selections.
