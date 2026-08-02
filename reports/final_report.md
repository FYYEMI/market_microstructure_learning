# Final Report

## 1. Research Design

This project evaluates whether simple limit-order-book imbalance features contain short-horizon directional information and whether those signals can inform simplified execution scheduling under visible-depth assumptions.

The dataset is BTC crypto limit-order-book snapshots from 2021-04-07 11:32:42.122161+00:00 to 2021-04-19 09:53:52.386544+00:00. The local raw source file contains 1,030,728 rows; the processed labelled dataset contains 1,030,698 rows after dropping rows without a 30-second future label. Quantity fields are dataset-provided quantity units and are not confirmed as BTC.

The headline prediction target is a three-class 30-second future mid-price direction label: `Up` if future return is greater than 1 bp, `Down` if lower than -1 bp, and `Flat` otherwise. The class counts are Up 395,120; Down 384,588; Flat 250,990.

The analysis uses chronological train / validation / test splits:

| Split | Date range |
|---|---|
| Train | 2021-04-07 11:32:42.122161+00:00 to 2021-04-15 19:59:40.182190+00:00 |
| Validation | 2021-04-15 19:59:41.182190+00:00 to 2021-04-17 14:56:36.119741+00:00 |
| Test | 2021-04-17 14:56:37.119741+00:00 to 2021-04-19 09:53:52.386544+00:00 |

The work is baseline-driven and intended for static historical analysis, not live execution or trading decisions.

## 2. Prediction Experiment

Features use only information available at or before the prediction timestamp: spread, relative spread, top-of-book imbalance, top-5 depth imbalance, displayed depth, and lagged mid-price returns. Future return is used only for target construction.

Methodological controls:

- Train / validation / test partitions are chronological 70% / 15% / 15% splits.
- Logistic Regression uses train-fitted median imputation and scaling inside a pipeline.
- XGBoost uses train-fitted median imputation; validation data is passed as an evaluation set, while reported headline metrics are from the test split.
- The test set is not used for model fitting or threshold selection in the retained public workflow.
- OBI decile boundaries for the displayed figure are estimated on the train sample and applied to the test sample.
- The label threshold is fixed at +/-1 bp for headline 30-second results.

The main 30-second test results are:

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| OBI rule | 0.4264 | 0.3713 | 0.3689 | 0.4253 |
| Logistic Regression | 0.4634 | 0.3759 | 0.3633 | 0.4394 |
| XGBoost | 0.4740 | 0.3774 | 0.3562 | 0.4422 |

The OBI rule remains competitive on macro F1. Logistic Regression improves accuracy versus the rule but does not improve test macro F1. XGBoost has the highest test accuracy but weaker macro F1 than both the rule and Logistic Regression. This supports a conservative interpretation: OBI has limited incremental information under this sample and label definition, and the nonlinear model does not produce a stable balanced-metric improvement.

Prediction-horizon robustness:

| Horizon | OBI rule macro F1 | Logistic macro F1 | Logistic minus rule |
|---:|---:|---:|---:|
| 10 seconds | 0.3832 | 0.3618 | -0.0214 |
| 30 seconds | 0.3688 | 0.3632 | -0.0056 |
| 60 seconds | 0.3493 | 0.3534 | +0.0041 |

## 3. Execution Experiment

The execution simulator uses visible top-5 depth walking. It does not model hidden liquidity, queue position, latency, fees, replenishment, or calibrated market impact.

The base comparison uses 20 rolling 30-minute windows, buy side, `target_qty = 500,000` dataset-provided quantity units, and 30 decision slices. The mean requested size is about 2.9532 times contemporaneous visible top-5 depth in these windows.

Compared schedules:

- immediate visible-depth walk
- static TWAP
- front-loaded schedule
- back-loaded schedule
- liquidity-weighted schedule / VWAP proxy
- fixed-seed random schedule
- OBI-adjusted TWAP

The VWAP-labelled schedule is a liquidity-weighted execution proxy based on displayed order-book conditions. It is not a reconstruction of market VWAP from actual traded-volume profiles.

At each schedule decision point, the simulator observes the current top-of-book and visible depth, calculates current spread/OBI/liquidity state, determines the child-order size from current and trailing information, walks the contemporaneous visible book, records fill and cost, and advances to the next timestamp.

| Schedule | Requested Qty | Filled Qty | Fill Rate | Mean Slippage bps | Penalized Cost 0 bps | Penalized Cost 5 bps | Penalized Cost 20 bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Immediate depth walk | 500,000.00 | 87,794.92 | 17.56% | 0.0914 | 5,570.22 | 11,617,562.74 | 46,453,540.29 |
| TWAP | 500,000.00 | 342,488.33 | 68.50% | 0.8422 | -234,642.98 | 4,197,843.49 | 17,495,302.92 |
| Front-loaded | 500,000.00 | 329,378.45 | 65.88% | 0.9626 | 366,537.59 | 5,168,028.79 | 19,572,502.38 |
| Back-loaded | 500,000.00 | 324,034.05 | 64.81% | 0.7866 | -1,042,851.59 | 3,908,027.82 | 18,760,666.03 |
| Liquidity-weighted schedule / VWAP proxy | 500,000.00 | 373,205.66 | 74.64% | 1.5062 | 1,155,226.31 | 4,721,735.78 | 15,421,264.21 |
| Random fixed seed | 500,000.00 | 330,354.26 | 66.07% | 0.6653 | -235,181.09 | 4,539,414.43 | 18,863,201.00 |
| OBI-adjusted TWAP | 500,000.00 | 320,830.29 | 64.17% | 0.3111 | -2,613,413.72 | 2,427,494.72 | 17,550,220.05 |

Slippage on filled quantity alone is not sufficient. OBI-adjusted TWAP has lower mean filled-quantity slippage than TWAP, but also lower fill rate. Under a 20 bps unfilled-quantity penalty, TWAP ranks slightly ahead of OBI-adjusted TWAP, while the liquidity-weighted schedule / VWAP proxy ranks first because it completes more quantity.

The penalty-adjusted cost is a sensitivity metric:

```text
execution_cost + unfilled_quantity * arrival_price * penalty_bps / 10000
```

It should not be interpreted as calibrated opportunity cost or full implementation shortfall.

## 4. Robustness

Regime-filter robustness after non-forward-looking schedule construction:

| Regime | Windows / Configurations | Retained Share | Mean OBI Minus TWAP Slippage bps | Median OBI Minus TWAP Slippage bps | OBI Beats TWAP Share | Mean OBI Fill Rate |
|---|---:|---:|---:|---:|---:|---:|
| Unfiltered | 20 | 100.00% | +0.6088 | +0.2244 | 30.00% | 58.02% |
| Filtered | 20 | 100.00% | +0.0420 | -0.0723 | 50.00% | 63.12% |

The regime filter uses expanding-window median spread/depth thresholds and an absolute OBI gate at each decision point. It reduces poor unconditional OBI scheduling behavior, but the filtered average result is close to zero rather than a robust improvement. Because this filter is exploratory and may reflect research selection, the result requires independent chronological validation.

Order-size robustness:

| Target Qty | Mean OBI Minus TWAP Slippage bps | OBI Beats TWAP Share | Mean OBI Fill Rate |
|---:|---:|---:|---:|
| 250,000 | +0.1047 | 50.00% | 69.54% |
| 500,000 | +0.0420 | 50.00% | 63.12% |
| 1,000,000 | -0.0645 | 70.00% | 53.56% |

Execution results are sensitive to requested size relative to visible top-5 depth. Apparent slippage improvements can be partly explained by lower fill rates.

## 5. Simplified Market-Making Extension

The market-making extension asks how symmetric and inventory-aware passive quoting behave under a simplified visible-order-book snapshot simulation. It is a mechanism study rather than a live strategy backtest.

Quotes are constructed from the current mid-price with a fixed half-spread. The inventory-aware variant shifts both quotes according to current inventory divided by the configured inventory limit: positive inventory shifts quotes lower to discourage additional buying and encourage selling; negative inventory shifts quotes higher. Fills use a deterministic one-next-snapshot touch proxy. The simulation tracks cash, inventory, fees, turnover, gross spread-capture proxy, inventory revaluation, marked PnL, and maximum inventory exposure.

Baseline results:

| Strategy | Total marked PnL | Fill rate | Trades | Avg abs inventory | Max abs inventory | Gross spread-capture proxy | Inventory revaluation PnL | Fees |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Symmetric | -24,793,739.01 | 21.64% | 2,095 | 6,660.10 | 12,500.00 | 5,992,820.40 | -27,831,170.00 | 2,955,389.41 |
| Inventory-aware | -18,375,265.54 | 21.49% | 2,149 | 1,864.30 | 6,500.00 | 5,977,278.53 | -21,321,217.50 | 3,031,326.56 |

The retained public simulation uses the first 25,000 chronological snapshots for notebook runtime, not to select a favorable PnL period. Inventory-aware quoting materially reduces average and maximum inventory exposure relative to symmetric quoting. Fill rate remains close to the symmetric baseline. Both variants produce negative marked PnL under the stated assumptions; the less negative inventory-aware result is mainly explained by a less adverse inventory revaluation component rather than proof of a positive spread-capture result.

Limitations are material: the data are LOB snapshots rather than message-level order events, and the simulation does not observe queue position, order priority, latency, hidden liquidity, cancellations, or true trade arrivals. Quantity units are dataset-provided units, and P&L should not be interpreted as real dollars.

## 6. Limitations

Data limitations:

- Crypto snapshots do not represent equities or institutional execution conditions.
- Snapshot data does not include full message-level order flow, queue position, hidden liquidity, latency, fees, or calibrated market impact.
- Top-5 visible depth is only a partial view of available liquidity.
- Quantity units are dataset-provided units and not confirmed as BTC.
- No trade prints or actual market-volume curve are available for true VWAP reconstruction.

Prediction limitations:

- Balanced-metric improvements are weak and horizon-dependent.
- The target is imbalanced and overlapping prediction horizons create serial dependence.
- The test period is limited and results may be dataset-specific.
- The analysis does not establish positive live trading performance.

Execution limitations:

- The simulator has no order priority, cancel/replace logic, hidden liquidity, depth replenishment, calibrated market impact, or venue fragmentation.
- Penalized execution cost is an assumption-driven sensitivity metric.
- The OBI-adjusted schedule is not optimal and is not consistently superior across penalty assumptions.

Interpretation limitations:

- This is personal research, not live trading, broker/exchange integration, investment advice, or production deployment.
- Results are not proof of future performance.

## 7. Conclusions

The retained results support a restrained set of conclusions:

1. OBI shows a weak short-horizon predictive relationship, but incremental model gains are limited and horizon-dependent.
2. Logistic Regression does not improve 30-second test macro F1 versus the OBI rule in the reproduced result.
3. XGBoost improves test accuracy but does not provide a stable balanced-metric improvement.
4. OBI-adjusted scheduling can reduce filled-quantity slippage in the base comparison, but it also lowers fill rate.
5. Completion penalties materially change schedule ranking, so execution results should be evaluated jointly with fill rate and unfilled quantity.
6. The market-making extension shows quote, fill, inventory, cash, and marked-PnL mechanics under a deterministic touch-based proxy, not positive live trading performance.

The repository demonstrates a structured research workflow: data standardization, chronologically aligned feature construction, chronological evaluation, visible-depth execution simulation, benchmark comparison, and robustness checks under simplified assumptions.
