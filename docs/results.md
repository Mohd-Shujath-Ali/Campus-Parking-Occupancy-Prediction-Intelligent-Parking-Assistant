# Results: model comparison, tuning, error analysis, explainability

All numbers are on the **chronological hold-out test period** (2026-10-18 16:20 → 2026-10-26 22:55, 8.3 days, 28,608 rows) of the **simulated** 56-day LPU dataset. Reproduce with the commands at the end. Read the limitations first: the data is generated, so these results show the methodology works, not how well it would forecast real traffic.

## 1. Validation design

| Split | Period | Rows | Notes |
|---|---|---|---|
| Train | 2026-09-01 → 10-10 | 135,192 | contains the exam week |
| Validation | 10-10 → 10-18 | 28,608 | contains the campus fest; used for stacking blend and probability calibration |
| Test | 10-18 → 10-26 | 28,608 | never used for fitting, tuning or threshold selection |

70/15/15 chronological split with a 1-hour buffer; features are built after splitting and per zone, so no lag or rolling window crosses a boundary. Hyper-parameter search uses `TimeSeriesSplit(5)` with a gap of 12 steps x 12 zones rows **inside the training period only**.

## 2. Model comparison (+15 min, test)

| Model | MAE (cars) | RMSE | R² | MAE in class-bell windows |
|---|---:|---:|---:|---:|
| Persistence (y(t)) | 6.63 | 10.12 | 0.979 | 15.33 |
| Seasonal mean (zone x weekday x hour) | 6.35 | 9.93 | 0.980 | 12.95 |
| Ridge | 6.13 | 9.16 | 0.983 | 15.29 |
| Random Forest | 4.16 | 5.68 | 0.994 | 7.34 |
| LightGBM | 4.41 | 6.01 | 0.993 | 6.89 |
| XGBoost (config defaults) | 4.21 | 5.71 | 0.993 | 6.32 |
| Stacking (RF + tuned XGB + LGBM -> non-negative Ridge) | 3.94 | 5.34 | 0.994 | 6.03 |
| **XGBoost (tuned)** | **3.68** | **4.95** | **0.995** | **5.05** |

Across horizons (test MAE): tuned XGBoost 3.68 / 3.71 / 3.79 for +15 / +30 / +60 min versus persistence 6.63 / 9.46 / 14.84 and the seasonal mean 6.35 / 6.73 / 6.36 (full table: `artifacts/model_comparison_full.csv`).

**Interpretation.**
* **Linear is not enough.** Ridge barely beats persistence: the target depends non-linearly on time of day and zone (bimodal occupancy, class-bell surges). Trees capture these interactions.
* **Boosting ≈ bagging on this data.** Random Forest and default XGBoost are within 0.06 cars at +15 min; XGBoost degrades slightly more slowly with horizon (+60 min: RF 5.10 vs default XGBoost 4.95).
* **The seasonal mean degrades far less with horizon than persistence** (6.36 vs 14.84 at +60 min) because simulated demand follows a fixed daily profile. The learned model's edge comes from combining that profile with the *current* state.
* **Stacking did not beat the best single model.** The blend (in an earlier run with default XGBoost the blend weights were RF 0.52, XGB 0.30, LGBM 0.19) mixes in weaker, highly correlated learners; with a tuned XGBoost it is 0.26 cars worse. Extra model diversity is only useful when the members make *different* errors.
* **Bell-window performance:** inside ±15 min of lecture bells persistence has 15.3 MAE; tuned XGBoost has 5.05 (**-67%**).

## 3. Hyper-parameter optimisation

Search space: trees 100-600, learning rate 0.01-0.2 (log), depth 3-10, min_child_weight 1-20, subsample, colsample, L1/L2 regularisation, gamma. 30 trials each, objective = mean CV MAE (+15 min).

| Search | Best CV MAE |
|---|---:|
| Bayesian (Optuna TPE) | **4.426** |
| Random search (same budget) | 4.501 |

Guided search found a better configuration than random sampling with the same budget (`artifacts/figures/tuning_history.png`). The winner (`configs/tuned_params.yaml`: depth 10, 550 trees, lr 0.03, strong `min_child_weight`/`gamma` regularisation, column subsampling 0.59) improved held-out error from **4.21 → 3.68** (test) and 4.43 → 4.06 (validation). The same parameters are reused for +30/+60 min (not tuned separately).

**Noise floor.** Re-generating the simulator with its noise term set to zero gives an oracle forecaster whose MAE on the same period is **3.38 cars**. The tuned model (3.68) is within ~0.3 cars of the best achievable on this data: the simulator is essentially solved, so further tuning on it is pointless and real data is the next lever.

## 4. Error analysis (tuned XGBoost, +15 min)

Full tables: `artifacts/error_analysis/*.csv`; figures: `artifacts/figures/error_*.png`, `forecast_example_day.png`.

* **Residuals are centred** (bias +0.08 cars; median |error| 2.8; 90th percentile 8.0; 99th 15.4). No systematic over/under-forecasting.
* **Error scales with lot size, not with difficulty.** Absolute MAE ranges from 2.3 (Central Mess 2, 110 spaces) to 6.1 (Main Parking North, 320 spaces), but as a share of capacity every zone sits at **1.9-2.3%**. The simulator's noise is proportional to capacity, so large lots look worse in absolute terms; judge zones by % of capacity.
* **Hardest hours are the morning ramp and class changes** (MAE ≈ 4.0-4.2 at 06-09 and 15-16 versus ≈ 2.8-3.0 overnight); persistence is worst at 07:00 (14.3) and 16:00 (12.0), exactly where the model gains most (-71% and -65%).
* **Bell windows are 38% harder** than steady state (5.03 vs 3.63), but 67% better than persistence there.
* **Weekends are easier** (3.34 vs 3.81) because demand is flatter and lower.
* **Near-full lots are slightly under-forecast** (bias -1.96 when actual ≥ 90% full): the model is conservative at capacity, which matters because those are the moments drivers care about.
* **Worst 20 predictions** (errors 24-33 cars) are all in the two largest lots. Eleven of the twenty are large 15-minute swings (e.g. +66 cars in 15 min at 08:40 on a Monday) that are not visible in the last hour of counts; six are isolated noise spikes and the rest sit in bell windows or near full capacity. These are information limits: a forecaster using only past counts cannot foresee an unscheduled surge.
* **Untested regimes.** The test period contains no exam or event days (they fall in train/validation), so performance during exam weeks and fests is **not measured**.

### Saturation classifier (P(occupancy ≥ 90% at +15 min))
* Imbalance 8.8 : 1. Uncalibrated, `scale_pos_weight` inflated probabilities: at the default 0.5 threshold precision was only **0.58** (recall 0.97).
* After isotonic calibration on the validation period the probabilities match observed frequencies (`classifier_diagnostics.png`): at 0.5, **precision 0.87, recall 0.75, F1 0.81, PR-AUC 0.90, ROC-AUC 0.99**. Calibration matters because the API shows this number and the ranker penalises lots above 0.70.

## 5. Explainability (SHAP, TreeExplainer)

Figures: `artifacts/figures/shap_*.png`; live endpoint: `GET /api/explain`.

**Global (mean |SHAP|, cars).** `vehicle_count` 34.0 ≫ `lag_15m` 10.5 ≈ `rolling_mean_30m` 10.3 > `occupancy_ratio` 2.5 > `hour_cos` 2.2 ≈ `hour_sin` 2.1 > the rest (zone identity 0.5; `is_class_transition` and the exam/event flags are below 0.5).
* The model is mostly **autoregressive**: it starts from the current level and recent trend (consistent with the 0.99 autocorrelation seen in the EDA).
* **Time of day acts as a correction term.** `hour_sin` (≈ +1 at 06:00, ≈ -1 at 18:00) adds up to +10 cars during the morning fill and subtracts 5-15 cars during the evening drain (`shap_dependence_hour_sin.png`).
* **Momentum:** `delta_15m` has a monotone effect (inflow of +5 cars over the last 15 min adds ~+4 to the forecast; outflow subtracts), small in absolute terms because the level is already carried by `vehicle_count`.
* Zone identity and capacity matter little: given their current count, the zones behave alike in the simulator.

**Local cases.**
* *Thu 08:50, Main Parking North:* 344 cars now, 301 fifteen minutes ago → forecast 346 (actual 362). `vehicle_count` contributes +149 versus the average-lot baseline of 66, the history features (`lag_15m`, `rolling_mean_30m`) add +44 and +36 (the lot has been busy all morning) and the morning time-of-day features add +13 and +6. The forecast is high because it is already full and still filling.
* *Sat 11:50, same lot:* 37 cars, falling → forecast 36 (actual 39). The low count pulls the forecast 21 cars below baseline, the falling lags subtract a further ~9 and no time-of-day feature is among the top contributors: the explanation is "empty and staying empty".

**Caveat.** SHAP shows what the model uses, not real-world causality. The flags `is_class_transition`, `is_exam_day` and `is_campus_event` carry ~zero attribution here because the current count already reflects those regimes by the time they matter.

## 6. Limitations
1. **Simulated data** (generator profiles + 2.5% Gaussian noise). The results validate the pipeline, not real-world accuracy; behaviour that the generator does not encode (weather, breakdowns, one-off events, driver behaviour) cannot be learned.
2. **Near-ceiling performance** (3.68 vs oracle 3.38) means additional modelling gains need real data.
3. **Test period lacks exam/event regimes**; their error is unmeasured.
4. **Single 8-day test window**; no confidence intervals. A rolling-origin evaluation over a full semester is needed.
5. Forecast is point-only; no prediction intervals yet.
6. Tuned parameters were searched for +15 min and reused for +30/+60.

## Reproduce
```bash
python -m data_pipeline.generator          # simulated dataset
python -m experiments.compare_models       # section 2 (~1 min)
python -m experiments.tune_xgboost         # section 3 (~10 min), writes configs/tuned_params.yaml
python -m models.train                     # fits and saves the deployed pipelines
python -m experiments.error_analysis       # section 4
python -m experiments.explain              # section 5
```

## 7. Prediction intervals and demand forecasting (Part C)

### 7.1 Prediction intervals (80%, conformalised quantile regression)
`models/intervals.py`: lower/upper quantile XGBoost fitted on train, widened by a conformal correction computed on the validation period. Coverage on the untouched test period (nominal 80%):

| Horizon | Test coverage | Mean width (cars) |
|---|---:|---:|
| +15 min | 0.808 | 14.0 |
| +30 min | 0.815 | 14.2 |
| +60 min | 0.826 | 14.9 |

Coverage is slightly above nominal, so the band is conservative rather than over-confident. The interval width barely grows with horizon because the simulator's noise (not forecast uncertainty) dominates. Exchangeability does not strictly hold for time series, which is why the realised test coverage is reported instead of being assumed. Served in `/api/predict` (`predicted_low/high`) and shown as the "80% Range" column in the admin console.

### 7.2 Demand forecasting (days ahead)
`models/demand.py`: calendar-only XGBoost (zone, time of day, weekday, exam/event flags; no live counts) predicting occupancy as a share of capacity, so it can forecast 1-14 days ahead and answer exam/event what-if questions via `GET /api/demand-forecast`. Test period, versus the zone x weekday x hour mean:

| Model | MAE (% of capacity) | RMSE | Peak hour within ±1 h | Peak-level MAE |
|---|---:|---:|---:|---:|
| Seasonal mean | 3.37 | 4.73 | 85.2% | 5.42 |
| **Calendar XGBoost** | **2.68** | **3.52** | **87.0%** | **4.44** |

The gain is modest (the simulator's demand is itself a fixed profile plus noise) and peak-hour identification is limited because peaks are flat-topped plateaus. The deployed model is refit on train + validation so the campus-event days, which fall in the validation period, are learned; exam and event effects therefore rest on one exam week and two event days and should be treated as indicative only. The what-if exam scenario raises weekday saturation hours for academic lots; the model has never seen real exam-week traffic.

### 7.3 Other Part C modules
* **Data collection** (`data_collection/`): SQLite store of per-camera counts; zone total = sum over cameras per 5-minute bucket; `POST /api/ingest` (optional `X-API-Key`) also feeds the rolling history so forecasts use real lags; `python -m data_collection.store --out data/processed/lpu_live.csv` exports real observations in the exact training schema.
* **Zone analysis** (`GET /api/zone-analysis/{zone}`): weekday/weekend profiles, peak hour, typical fill/drain hour, saturation share. Vehicle turnover is intentionally not reported: counts only give net change, and 5-minute counts are noisy, so a churn figure would mostly measure noise.
* **Heatmaps** (`GET /api/heatmap`): historical zone x hour (weekday/weekend), one zone's weekday x hour, and tomorrow's forecast.
* **Admin alerts**: zones forecast at ≥ 90%, saturation probability ≥ 0.70, or stale cameras.
* All historical views are labelled `data_source: simulated` until real observations replace the generator.
