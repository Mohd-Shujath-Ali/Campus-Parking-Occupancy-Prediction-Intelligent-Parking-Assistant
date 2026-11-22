# Campus Parking Occupancy Prediction & Intelligent Parking Assistant (Project 32)
## School of Computer Science & Engineering — Lovely Professional University (LPU)

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-v8%20%2F%20v26-orange.svg)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.115-009688.svg)](https://fastapi.tiangolo.com)
[![HTML5/Tailwind](https://img.shields.io/badge/Frontend-HTML5%20%7C%20TailwindCSS%20%7C%20Chart.js-38B2AC.svg)](https://tailwindcss.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.1-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-TreeSHAP-brightgreen.svg)](https://shap.readthedocs.io/)

---

## 1. Project Overview

This project delivers an end-to-end Machine Learning, Computer Vision, and Decision Support system for parking management across the 600-acre Lovely Professional University (LPU) campus in Phagwara, Punjab.

Unlike traditional marked-stall sensor systems, this architecture addresses **open, unstructured campus grounds** using:
1. **Perception**: Fine-tuned YOLO (`best.pt`, mAP50=91.6%) and Pretrained YOLO (`yolo26l.pt`) with Ground-Contact Point Jordan Ray-Casting Point-in-Polygon (PIP), cross-camera Global Spatial NMS ($d < 2.5\text{m}$), and OpenCV pre-inference quality diagnostics (blur, underexposure, glare); optional in-browser YOLO via WebGPU/WASM.
2. **Zero-Leakage Time-Series Pipeline**: Multi-horizon forecasting ($+15, +30, +60$ min, interpolated to any arrival time) with a chronological 70/15/15 split (1 h purge buffer) and 41 causal features: sin/cos time, zone identity, lags ($t-15 \dots t-60$), velocity, shifted rolling statistics, a leave-one-week-out **normal-day profile** per zone and time of week, and what is already known about the *arrival* time (events, exams, holidays, distance to a lecture bell). Calendar data comes from the admin-managed campus calendar. All preprocessing lives in one persisted scikit-learn `Pipeline`/`ColumnTransformer`; hyperparameters are in `configs/model_config.yaml` (Optuna-tuned overrides in `configs/tuned_params.yaml`). Design, evidence and limits: [`docs/FORECASTING.md`](docs/FORECASTING.md).
3. **Multi-Model Comparative Benchmarking**: Persistence, normal-day-profile and zone-aware Seasonal Mean baselines, Ridge, Random Forest (bagging) and XGBoost (boosting), compared with `experiments/forecast_ablation.py`; boosting wins and a bagging blend adds nothing. Hyperparameters were Bayesian-tuned (Optuna TPE) on the earlier dataset with `TimeSeriesSplit` cross-validation; an optional `GridSearchCV` runner is kept in `models/train.py --grid-search`.
4. **Dual-Head Architecture**: Multi-horizon count regressor plus a Saturation Classifier ($P(\text{Occ} \ge 90\%)$) using `scale_pos_weight` and isotonic probability calibration on the validation period.
5. **Uncertainty & Demand Forecasting**: calibrated 80% ranges (held-out coverage 0.79-0.80, 0.82-0.89 during events), the **chance of a free slot on arrival** (Brier 0.012-0.016 vs 0.056 for the base rate), and a calendar-only demand forecaster for days-ahead planning that applies the campus calendar automatically.
6. **Explainable AI (SHAP)**: Global beeswarm and mean $|SHAP|$ rankings, dependence plots, local waterfall traces, and a live per-prediction breakdown at `GET /api/explain`.
7. **Error Analysis & Diagnostics**: Residuals vs. fitted, Q-Q plots, per-zone / per-hour / per-regime error tables, worst-case review and classifier calibration.
8. **Anti-Herding Traffic Dispersion**: A softmax lead-zone selector among near-tied lots plus a real-time `InFlightDispatchTracker` (15-minute TTL, 35% capacity cap) fed by `POST /api/navigate`.
9. **Data Collection**: SQLite store of per-camera counts, summed per zone, with `POST /api/ingest` and an exporter that writes real observations in the training schema.
10. **Full-Stack Web Microservice**: Asynchronous **FastAPI** backend with SlowAPI rate limiting, serving responsive multi-page **TailwindCSS & Chart.js** interfaces (admin alerts, heatmaps, 7-day demand outlook, zone usage profiles) and a static build for Render.

---

## 2. Monitored LPU Campus Zones (Empirical Satellite Survey)

Calibrated against aerial surveys from `collab_content_backup/LPU PARKINGz`:

| Zone ID | Name | Category | Capacity | Key Adjacent Buildings |
| :--- | :--- | :--- | :---: | :--- |
| `ZONE_MAINGATE` | LPU Main Gate Parking | Campus Entrance & Transit | 240 | GT Road, Main Security Post |
| `ZONE_BLOCK1` | Block 1 Parking | Academic & Administrative Core | 140 | Block 1, Baldevraj Mittal Auditorium, Uni Health Center |
| `ZONE_ARCH` | Architecture Block Parking | Design & Applied Arts | 120 | Architecture Block, Product Block, Hospital |
| `ZONE_MBFS` | MB Food Square Parking | Dining & Social Hub | 200 | MB Food Square, Shri Baldev Raj Mittal Hospital |
| `ZONE_LAW` | Law Block Parking | School of Law & Management | 130 | School of Law, Management Block |
| `ZONE_GH5` | GH 5 Parking | Residential (Hostels) | 160 | GH-5 Girls Hostel, Boys Hostel Complex |
| `ZONE_MAINPARK_NORTH` | LPU Main Parking (North) | Computing & Admissions | 320 | Division of Admissions, Block 26 (School of CS) |
| `ZONE_MAINPARK_SOUTH` | LPU Main Parking (South) | Computing & Innovation | 300 | School of CSE, Innovation Studio Block 39 |
| `ZONE_CENTRAL_MESS2` | Central Mess 2 Parking | Dining & Residential | 110 | 41-D Block, Lovely Bake Studio, S Mess Rd |
| `ZONE_BH2` | Boys Hostel 2 Parking | Residential & Stadium | 135 | LPU Boys Hostel 2, XL Arena, Indoor Stadium |
| `ZONE_SPORTS` | LPU Sports Ground Parking | Athletics & Sports | 145 | Basketball Courts, Kabaddi Ground |
| `ZONE_BLOCK57` | LPU Block 57 Parking | Agriculture & Mechanical | 185 | School of Agriculture (Block 57), Mechanical (Block 55) |

---

## 3. Machine Learning Benchmark Results (Holdout Test Split)

> **Data caveat.** The occupancy series in `data/processed/lpu_timeseries.csv` is **simulated** by `data_pipeline/generator.py` (12 zones, 5-minute cadence, 18 weeks, with exam windows, holidays and events in every split) until real detector counts are logged. These results show that the pipeline works, not how well it would forecast real LPU traffic; expect real accuracy to be lower. The app labels itself "demo data" until a camera reports.
>
> Prefer the skill column to $R^2$: occupancy is strongly autocorrelated, so even "same as now" scores $R^2 \approx 0.97$ at +15 min.

| Model Architecture | Forecast Horizon | Data Partition | MAE (Cars) | RMSE (Cars) | $R^2$ Score | Skill vs Persistence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** ("same as now") | +15 / +30 / +60 min | Holdout Test | 5.555 / 8.912 / 14.683 | 12.40 / 19.00 / 29.37 | 0.969 / 0.928 / 0.828 | — |
| **Normal-day Profile** (no live data) | +15 / +30 / +60 min | Holdout Test | 8.326 / 8.325 / 8.325 | 17.43 | 0.939 | −50% / +7% / +43% |
| **Seasonal Historical Mean** (zone × weekday × hour) | +15 min | Holdout Test | 15.009 | 31.57 | 0.801 | −170% |
| **Ridge Regression ($L_2$)** | +15 / +30 / +60 min | Holdout Test | 4.483 / 6.088 / 8.216 | 10.09 / 13.34 / 17.09 | 0.980 / 0.964 / 0.942 | 19% / 32% / 44% |
| **Random Forest** (bagging) | +15 / +30 / +60 min | Holdout Test | 3.804 / 4.614 / 5.553 | 6.09 / 7.09 / 8.44 | 0.993 / 0.990 / 0.986 | 32% / 48% / 62% |
| **Tuned XGBoost** (boosting) | +15 min | Validation | 3.542 | 5.996 | 0.9930 | 36% |
| **Tuned XGBoost** (boosting) | +15 / +30 / +60 min | Holdout Test | **3.485 / 4.151 / 4.776** | **5.19 / 6.18 / 6.93** | **0.995 / 0.992 / 0.990** | **37% / 53% / 67%** |
| **Saturation Classifier** (isotonic-calibrated) | +15 min | Holdout Test | — | — | — | **PR-AUC = 0.9725** (Recall 90.3%, Precision 0.91) |

At lecture-bell windows (+15 min) XGBoost's error is 4.4 vs 11.3 for persistence. By kind of day at +60 min: normal weekday 5.3, weekend 4.0, holiday 4.2, exam day 4.6, **event 8.9** (persistence 20.5). Following the recommendation finds a free slot **99.7-99.9%** of the time on the held-out replay, against 91.9% for walking to the nearest ground (see [`docs/FORECASTING.md`](docs/FORECASTING.md)).

Raw tables: `artifacts/model_comparison.csv`, `artifacts/regime_metrics.csv`, `artifacts/ablation.csv`, `artifacts/recommendation_eval.csv`. [`docs/results.md`](docs/results.md) predates this version (earlier 56-day dataset); treat its numbers as historical.

---

## 4. Quick Start & Execution Guide

### Prerequisites & Installation
```bash
# Clone the repository
git clone https://github.com/Mohd-Shujath-Ali/Campus-Parking-Occupancy-Prediction-Intelligent-Parking-Assistant.git
cd Campus-Parking-Occupancy-Prediction-Intelligent-Parking-Assistant

# Install pinned dependencies
pip install -r requirements.txt
```

### 1. Run End-to-End Training & Diagnostics
```bash
python -m models.train
```
*Outputs: loads `data/processed/lpu_timeseries.csv` (regenerate with `python -m data_pipeline.generator`), fits the baselines, Ridge, Random Forest and tuned XGBoost (multi-horizon), the calibrated saturation classifier, conformal intervals and the demand forecaster; writes error-analysis plots (`residuals_vs_fitted.png`, `error_distribution_qq.png`, `error_by_zone_boxplot.png`), `artifacts/model_comparison.csv` and the serialised pipelines in `artifacts/pipelines/`. Add `--grid-search` for the quick `GridSearchCV` runner.*

### 2. Generate Explainable AI (SHAP) Interpretability Suite
```bash
python -m explainability.shap_analysis
```
*Outputs: Generates TreeSHAP summary beeswarm, feature importance bar charts, and local rush-hour / steady-state decision waterfalls to `artifacts/figures/`.*

### 3. Reproduce the experiments
```bash
python -m experiments.forecast_ablation              # old vs new features, direct vs profile-relative, bagging vs boosting
python -m experiments.interval_comparison            # CQR vs residual-distribution intervals
python -m experiments.replay_recommendations --tune  # choose ranker weights on the validation period
python -m experiments.replay_recommendations         # does following the recommendation find a slot? (test period)
python -m models.retrain --dry-run                   # champion/challenger on real counts, once cameras report
# Earlier experiments (written for the previous dataset and feature set; may need adapting):
python -m experiments.compare_models
python -m experiments.tune_xgboost
python -m experiments.error_analysis
python -m experiments.explain
```

### 4. Launch FastAPI Web Application & API Server
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Everyone signs in at `/login` (see [`docs/ACCESS_CONTROL.md`](docs/ACCESS_CONTROL.md)); each role sees only its pages.
- **Driver Assistant**: `http://localhost:8000/` (arrival-time slider, chance of a free slot per ground, data-source labels, computed arrival tips, anti-herding dispatch).
- **Campus Calendar** (admin): `http://localhost:8000/calendar` (events with venue and expected vehicles, exam windows, holidays; forecasts use them immediately).
- **Model Health** (admin): `http://localhost:8000/model` (model version and training data, where each ground's count comes from, live accuracy and drift).
- **Admin Command Center**: `http://localhost:8000/admin` (Real-time occupancy charts with 80% forecast ranges, operational alerts, camera sensor telemetry, invariant stream detection).
- **Vision & ROI Studio**: `http://localhost:8000/vision` (Live webcam stream, interactive polygon/box ROI drawing, dual-model YOLO benchmarking).
- **Model Analytics & XAI**: `http://localhost:8000/analytics` (Benchmark comparisons, SHAP attribution tabs, residual diagnostics, zone × hour demand heatmaps, 7-day demand outlook with exam/event scenarios).
- **Campus Topology & Heatmap**: `http://localhost:8000/zones` (Live 12-zone congestion heatmap grid, grounds directory and per-zone usage profiles).
- **Interactive REST Swagger**: `http://localhost:8000/docs`.

### 5. Deploy
See [`docs/deployment.md`](docs/deployment.md): a Render blueprint (`render.yaml`, deploys the `testing` branch from the lean `requirements-render.txt`) or Docker.

### 6. Run Test Suite
```bash
pytest tests/ -v
```
*Runs the automated unit and integration tests (API endpoints, pipelines, intervals, demand forecast, data collection, anti-herding dispatch, history/stream-freeze logic, computer-vision filters). `test_detect_image_endpoint` needs `ultralytics`/`torch` installed.*

---

## 5. REST API Documentation

- `GET /api/health`, `GET /api/zones`: system status and metadata for all 12 zones.
- `POST /api/predict`: occupancy forecast for a zone at any horizon from 5 to 60 min (interpolated between the trained +15/+30/+60) with an 80% interval (`predicted_low/high`), `p_free` (chance of a free slot), `lag_source` (`observed|partial|estimated`) and `stream_frozen`. Exam/bell flags come from the campus calendar unless overridden.
- `GET /api/recommendations?destination=Block_26_CSE&horizon_minutes=30`: ranked parking grounds with `p_free`, 80% range, `confidence` and `count_source` per ground, plus `data_mode` and today's `conditions` (destination matching is case/space tolerant); `POST /api/navigate` registers an in-flight driver dispatch.
- `GET /api/snapshot`: every ground right now (live camera count, or labelled demo/estimate) plus the +15 min forecast, ranges and server-side alerts; `GET /api/metrics`: benchmark table and classifier metrics.
- `GET|POST|DELETE /api/calendar`, `GET /api/calendar/context`, `POST /api/calendar/preview`: the campus calendar (writes are admin-only).
- `GET /api/model/status`, `POST /api/model/reload` (admin): model version, data sources, live accuracy and drift; load a newly trained model without a restart.
- `GET /api/explain?zone_id=ZONE_BLOCK1&current_count=80`: SHAP breakdown of a forecast (base value + per-feature effect in cars); `GET /api/shap-figure/{name}`: generated diagnostic figures.
- `GET /api/demand-forecast?days=7&zone_id=ZONE_MAINPARK_NORTH&exam_dates=2026-11-23`: days-ahead peak time/level per zone with exam/event what-if.
- `GET /api/zone-analysis/{zone_id}` and `GET /api/heatmap?kind=zone_hour|zone_week|forecast`: usage profiles and heatmap matrices.
- `POST /api/ingest` and `GET /api/observations/stats`: collect real camera counts (optional `X-API-Key`, set `INGEST_API_KEY`).
- `POST /api/detect-image`, `POST /api/detect-frame`: YOLO detection (rate limited).

---

## 6. Data Sources, Serving Notes & Limitations

**Occupancy time series.** Simulated (see the caveat in Section 3). Real counts replace it through `POST /api/ingest`; `python -m models.retrain` then retrains on them and promotes the new model only if it is better. `python -m scripts.simulate_ingest` feeds fake counts to a running server to try the loop.

**Vision data.** The YOLO training images in `collab_content_backup/LPU---Parking-1` come from a Roboflow export (licence CC BY 4.0, see its `README.roboflow.txt`). The image file names follow the timestamp pattern of the public PKLot dataset, so confirm the original source and cite it before submission.

**Serving.** `/api/recommendations` and `/api/snapshot` use the latest ingested count per ground (rolling history in `models/history.py`, reloaded from the observation store at startup). A reading up to 7 min old is live, up to 30 min stale (lower confidence); with no usable reading a ground shows the normal-day estimate. A simulated demo feed is used only while no camera has ever reported, and is labelled everywhere it appears. A frozen feed (6 identical readings over 10+ minutes) sets `stream_frozen`.

**Known limitations.** The data is synthetic, so accuracy on real traffic is unknown and will be lower. Event calibration rests on three validation events. XGBoost hyper-parameters were tuned on the earlier dataset and reused. Counts are per ground, not per slot. Weather is not modelled. The static SHAP figures in `artifacts/figures/` predate the current model (`GET /api/explain` is current). The notebooks in `notebooks/` use an older, simpler pipeline. Full list: [`docs/FORECASTING.md`](docs/FORECASTING.md#limitations).

**Not yet done:** real (logged) occupancy data actually collected, a server-side camera worker that turns RTSP/HTTP streams into `/api/ingest` counts (camera sources are stored but only the browser Vision page produces counts today), public deployment of the API, slot-level occupancy.

---

## 7. Jupyter Research Notebooks

The `notebooks/` directory contains structured, reproducible experiments aligned with the course rubrics:
- `notebooks/01_exploratory_data_analysis.ipynb`: Data quality checks, ADF stationarity tests, ACF/PACF lag selection, bell surge distributions, exam period density shifts, saturation class imbalance, and correlation heatmaps.
- `notebooks/02_model_training_and_tuning.ipynb`: Chronological train/val/test splitting, baseline vs. ML model comparisons, TimeSeriesSplit GridSearchCV curves, surge inflection error breakdown, and dual-head classification evaluation.
- `notebooks/01_eda.ipynb`: a second, independent EDA covering data quality, distributions, daily/weekly patterns, special regimes, autocorrelation, class imbalance and split composition.
- `notebooks/smote_experiment.ipynb`: Oversampling and imbalance handling experiments.

---

## 8. Team Contributions

- **Mohammed Shujath Ali (12404599)**: Forecasting architecture, TimeSeriesSplit hyperparameter tuning, TreeSHAP interpretability, full-stack FastAPI microservice, TailwindCSS/Chart.js frontends, anti-herding dispatch tracker, and project coordination.
- **SREEVYAS CS (12400848)**: Computer vision ROI polygon management, Ground contact point geometry, cross-camera Spatial NMS deduplication, image degradation diagnostics, test suite development.
- **Dr. Mohit Arora (Faculty Supervisor)**: Project mentorship, technical guidance, and academic evaluation.
