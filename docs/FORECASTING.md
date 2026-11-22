# Forecasting and recommendations

How real-time counts, past patterns and the campus calendar become a parking recommendation, why it is built this
way, and what the evidence says. All accuracy numbers are from the untouched hold-out test period of the
**synthetic** dataset (see [Limitations](#limitations)); reproduce them with the commands at the end.

## The loop

```
camera count per zone (POST /api/ingest, every 5 min)
        |                       campus calendar (admin: events, exams, holidays) + bell schedule
        v                                  |
 rolling history  ------------------>  feature builder (41 features)  <---- normal-day profile (zone x time of week)
                                           |
                       XGBoost x 3 horizons (+15 / +30 / +60 min)
                                           |
              calibrated uncertainty: 80% range, chance of a free slot on arrival
                                           |
        interpolate to the driver's arrival time (5-120 min)  ->  ParkingRanker  ->  recommendation
```

Per ground, `models/live.py` picks the count to start from:

| Source | When | Shown as |
|---|---|---|
| **camera** | a count ingested within 7 min (live) or 30 min (stale, lower confidence) | no badge |
| **simulated** | only while *no camera has ever reported* (demo) | "Demo" badge and a page banner |
| **profile** | real data has been seen, but this ground has no usable reading | "Estimate" badge, low confidence |

Once any real count has been ingested the demo feed is never used again, so a camera outage cannot silently turn
into invented numbers. A reading that is `age` minutes old forecasts `horizon + age` minutes ahead of that reading.

## What the model is, and why

**Direct XGBoost, one model per horizon**, trained on a chronological 70/15/15 split with a 1 h purge. Each model sees:

* the live state: count, lags (15-60 min), velocity, rolling mean/std
* the **normal-day profile**: how full this ground normally is at this time of week, and how far today deviates from it
  (the "past data by timing"; built from training data only, leave-one-week-out for the rows it trains on)
* **what is already known about the arrival time**: profile, exam, holiday, event load and distance to a lecture bell
  at *t + horizon*, because calendar and timetable are known in advance
* calendar flags for now: weekday/weekend, holiday, exam day, bell window, event load

### Bagging vs boosting: measured, not assumed

`python -m experiments.forecast_ablation` (MAE in vehicles on the hold-out test; lower is better):

| Model | +15 min | +30 min | +60 min |
|---|---|---|---|
| Persistence ("same as now") | 5.56 | 8.91 | 14.68 |
| Normal-day profile only | 8.33 | 8.32 | 8.32 |
| XGBoost, previous features | 3.61 | 4.41 | 5.37 |
| **XGBoost, new features (shipped)** | **3.48** | **4.15** | **4.78** |
| XGBoost, profile-relative target | 3.52 | 4.19 | 5.00 |
| Random Forest (bagging), profile-relative | 4.20 | 4.71 | 5.21 |
| Blend: XGBoost + Random Forest | 3.54 | 4.22 | 5.01 |

* **Boosting wins.** Random Forest is worse at every horizon, and blending it in adds nothing (the blend put 77-91% of its
  weight on XGBoost). Bagging stays in the benchmark table for comparison only.
* **The new features matter most at longer horizons**: 11% lower error at +60 min, 12% lower during events, and
  holidays go from 6.56 to 4.22 because the old model had no way to know about them.
* **A tested idea that did not help**: predicting the *deviation from the profile* instead of the count was slightly
  worse, so it is not offered. The profile reaches the model as features instead.
* Rejected without testing: LSTM/transformers (12 series, little data, hard to explain) and LightGBM (new dependency, no
  expected gain over XGBoost here).

Shipped model, hold-out test: error is **37% / 53% / 67%** lower than "same as now" at +15 / +30 / +60 min. By kind of day
at +60 min (vehicles): normal weekday 5.3, weekend 4.0, holiday 4.2, exam day 4.6, **event 8.9** (persistence: 20.5).
Events are the hardest case: each synthetic event turns out 0.6-1.4x its declared size, which no model can predict.

## Uncertainty and the "will there be a slot?" number

A point forecast is not what a driver needs. `models/uncertainty.py` keeps the distribution of
(actual - forecast) measured on the validation period, per horizon and per situation (how full the ground is, near a
bell, event on). From it:

* **P(free slot on arrival)**: chance that at least `max(2, 3% of capacity)` slots are free
* **80% range** for the vehicle count
* P(>= 90% full): the calibrated classifier at +15 min, the same distribution at +30/+60

On the hold-out the 80% ranges contain **79.8% / 79.3% / 80.1%** of outcomes (81.8% / 81.5% / 89.3% during events), and the
free-slot probability has Brier scores of 0.012-0.016 against 0.056 for always guessing the base rate. Feature-aware CQR
intervals were compared (`experiments/interval_comparison.py`): same coverage but **30-46% wider** bands, so the simpler
method is used and CQR stays opt-in (`python -m models.train --cqr`).

## Arrival times hours ahead (9 AM, 1 PM, 5-8 PM buttons)

Drivers can pick a clock time (`arrive_at=17:00`, the next occurrence in campus time, or a date-time up to 7 days ahead).
The live models stop at +60 min. Beyond that the forecast is the **calendar model at the arrival time** (exams, holidays,
events, time of week) plus whatever the live forecast says is unusual today, fading with a 180-minute time constant:

    forecast = calendar(arrival) + (live +60 forecast - calendar(reading + 60)) x exp(-(h - 60) / 180)

`python -m experiments.far_horizon` (fade constant chosen on validation, MAE in vehicles on test):

| Method | 1.5 h | 2 h | 4 h | 8 h | 12 h |
|---|---|---|---|---|---|
| Previous code (decayed toward the normal level at reading + 60 min) | 9.5 | 14.1 | 31.5 | 59.6 | 79.8 |
| Normal-day profile at arrival | 8.4 | 8.4 | 8.4 | 8.4 | 8.4 |
| Calendar model at arrival | 7.2 | 7.2 | 7.3 | 7.3 | 7.3 |
| **Blend (shipped)** | **6.1** | **6.8** | **7.1** | **7.2** | **7.3** |

The previous code anchored on the wrong time: a 5 PM forecast made at 9 AM looked like 10 AM. On exam, holiday and event arrival
times the blend scores 7.2-8.3 against 11.0 for the profile; at 2-3 h ahead on those days the calendar model alone is slightly
better (7.8 vs 8.3). The calendar model's 80% ranges cover 78.5% of outcomes (86.8% during events), and its free-slot Brier
score is 0.025 against 0.056 for the base rate: less sharp than the live model, as expected that far out. Each answer says
which basis it used (`forecast_basis`: live / blended / calendar), and a fresh camera no longer earns "high" confidence for
an arrival many hours away.

## The campus calendar

Admins enter exam windows, holidays and events on **/calendar**. An event is a venue block, a time window and the extra
vehicles expected; `features/calendar.py` spreads those vehicles over the grounds by walking distance from the venue
(Block 26 with 600 extra vehicles loads Main Parking South and North by roughly +50% each). Entries change forecasts on the next
request, appear in the 7-day demand outlook, and are what retraining uses as exam/holiday/event flags for real data.
Bells are `bell_times_minutes_of_day` in `configs/model_config.yaml`.

## Recommendations

Grounds are ranked on the calibrated chance of a free slot on arrival, adjusted for drivers already heading there, and
walking distance. Weights (`availability 0.25, distance 0.7, p_free_blend 1.0`) were **chosen on the validation period**;
the test period was used only to report.

`python -m experiments.replay_recommendations` replays the test period (912 decision times x 8 destinations x 3 arrival
times). Share of drivers who found a free slot on arrival, and average walk:

| Policy | Found a slot (+15 / +30 / +60) | Avg walk | Expected cost* |
|---|---|---|---|
| Walk to the nearest ground | 91.9% | 34 m | 74 m |
| Nearest ground with space *right now* | 97.8 / 96.5 / 94.8% | 54 m | 65 / 71 / 80 m |
| **Recommender** | **99.9 / 99.7 / 99.7%** | **71-73 m** | **73 m** |
| Oracle (upper bound) | 100% | 54 m | 54 m |

\*walk + 500 m x failure rate. Honest reading: at +15 min, looking at the lot right now is just as good; the recommender
pulls ahead at longer arrival times and on event days (98.3% vs 94.1% at +30 min). The anti-herding randomiser costs about
8 m of walk; its benefit (spreading demand) cannot show in a replay where simulated drivers do not take slots.

Arrival tips are computed from the forecast ("arriving ~30 min later raises your chance from 40% to 85%"), not canned.

## Monitoring and retraining

* Every camera-based forecast is logged. When the real count for its target time arrives it is scored: live MAE, bias,
  80%-range hit rate and free-slot Brier per horizon, with a drift warning when live MAE exceeds `drift_mae_ratio` (1.5x)
  the training MAE. See **/model** (admin) and `GET /api/model/status`.
* `python -m models.retrain [--dry-run]` trains a challenger on real counts and promotes it **only if** it beats the
  current model by >= 2% MAE at +15 and +60 min on the latest real week while keeping interval coverage in 72-88%. Needs
  >= 14 days from >= 6 grounds; below 56 days real rows are mixed with the synthetic block (real weighted 5x), after that
  synthetic is dropped. The old model is archived (`artifacts/archive/`) and decisions are logged in `artifacts/registry.json`.
  After promoting, `POST /api/model/reload` (admin) loads it without a restart.
* `python -m scripts.simulate_ingest` feeds synthetic "camera" counts to a running server to try the loop end to end.

## Commands

```bash
python -m models.train                          # full training + benchmark tables (artifacts/*.csv, model_meta.json)
python -m models.train --fast                   # smaller models for development
python -m experiments.forecast_ablation         # the model-choice table above
python -m experiments.interval_comparison       # CQR vs residual-distribution intervals
python -m experiments.far_horizon               # how to forecast arrival times hours ahead
python -m models.train --demand-only            # refit just the calendar model and its uncertainty
python -m experiments.replay_recommendations --tune   # choose ranker weights on the validation period
python -m experiments.replay_recommendations          # report on the test period
python -m models.retrain --dry-run              # when real counts exist
```

## Limitations

* **The training data is synthetic.** Models learn `data_pipeline/generator.py`, not LPU. The numbers above are a
  pipeline check, and real accuracy will be lower. The UI says "demo data" until a camera reports, `/model` shows
  `trained on: SYNTHETIC`, and live accuracy only becomes meaningful once real counts flow.
* Event calibration rests on **three validation events**; event intervals are slightly conservative at +60 min (89% vs 80%).
* XGBoost hyper-parameters (`configs/tuned_params.yaml`) were tuned on the earlier dataset and reused, not re-tuned.
* Counts are per ground, not per slot. A camera's region counts vehicles in its polygon; partial reporting by one of
  several cameras covering a ground under-counts that bucket.
* Weather is not modelled; the generator's hidden rain days show up as unexplained error.
* Forecast history (`models/history.py`) is per process: run one worker (the Docker default) or share the observation store.
* The SHAP figures under `artifacts/figures/shap_*.png` were generated for the earlier model; `GET /api/explain`
  (native TreeSHAP) is always current. Regenerate with `python -m experiments.explain` (needs `shap`).
