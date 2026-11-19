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
2. **Zero-Leakage Time-Series Pipeline**: Multi-horizon forecasting ($+15, +30, +60$ min) with a chronological 70/15/15 split (1 h purge buffer) and 22 causal features: sin/cos time, zone identity, event/exam flags, lags ($t-15 \dots t-60$), velocity and shifted rolling statistics. All preprocessing lives in one persisted scikit-learn `Pipeline`/`ColumnTransformer`; hyperparameters are in `configs/model_config.yaml` (Optuna-tuned overrides in `configs/tuned_params.yaml`).
3. **Multi-Model Comparative Benchmarking**: Persistence and zone-aware Seasonal Mean baselines, Ridge, Random Forest, LightGBM, a leak-free blend stack, and XGBoost tuned by Bayesian search (Optuna TPE vs random search) with `TimeSeriesSplit` cross-validation; an optional `GridSearchCV` runner is kept in `models/train.py --grid-search`.
4. **Dual-Head Architecture**: Multi-horizon count regressor plus a Saturation Classifier ($P(\text{Occ} \ge 90\%)$) using `scale_pos_weight` and isotonic probability calibration on the validation period.
5. **Uncertainty & Demand Forecasting**: 80% conformal prediction intervals (held-out coverage 0.81-0.82) and a calendar-only demand forecaster for days-ahead planning with exam/event what-if scenarios.
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

> **Data caveat.** The occupancy series in `data/processed/lpu_timeseries.csv` is **simulated** by `data_pipeline/generator.py` (12 zones, 5-minute cadence, 56 days) until real detector counts are logged. These results show that the pipeline works, not how well it would forecast real LPU traffic. A noise-free oracle on this simulator scores 3.38 cars MAE, so the tuned model is within ~0.3 cars of the ceiling.

| Model Architecture | Forecast Horizon | Data Partition | MAE (Cars) | RMSE (Cars) | $R^2$ Score | Surge Inflection MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | +15 min | Holdout Test | 6.631 | 10.116 | 0.9794 | 15.329 |
| **Seasonal Historical Mean** (zone × weekday × hour) | +15 min | Holdout Test | 6.354 | 9.925 | 0.9802 | — |
| **Ridge Regression ($L_2$)** | +15 min | Holdout Test | 6.129 | 9.157 | 0.9831 | 15.290 |
| **Random Forest Ensemble** | +15 min | Holdout Test | 4.163 | 5.673 | 0.9935 | 7.368 |
| **Tuned XGBoost** (Optuna) | +15 min | Validation | 4.049 | 6.279 | 0.9918 | — |
| **Tuned XGBoost** (Optuna) | +15 min | Holdout Test | **3.677** | **4.945** | **0.9951** | **5.027 (−67% vs Pers.)** |
| **Tuned XGBoost** (Optuna) | +30 min | Holdout Test | 3.717 | 4.990 | 0.9950 | — |
| **Tuned XGBoost** (Optuna) | +60 min | Holdout Test | 3.783 | 5.093 | 0.9948 | — |
| **Saturation Classifier** (isotonic-calibrated) | +15 min | Holdout Test | — | — | — | **PR-AUC = 0.9041** (Recall: 75.0%, Precision 0.87) |

Extended comparison (LightGBM, a chronological blend stack, default-vs-tuned XGBoost), Bayesian-vs-random search, error analysis and SHAP interpretation are in [`docs/results.md`](docs/results.md); raw tables are in `artifacts/model_comparison.csv` and `artifacts/model_comparison_full.csv`.

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

### 3. Reproduce the extended experiments
```bash
python -m experiments.compare_models   # Ridge / RF / XGBoost / LightGBM / stacking vs baselines
python -m experiments.tune_xgboost     # Optuna TPE vs random search -> configs/tuned_params.yaml
python -m experiments.error_analysis   # per-zone / hour / regime errors, worst cases, classifier calibration
python -m experiments.explain          # SHAP dependence and local plots
```

### 4. Launch FastAPI Web Application & API Server
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Driver Assistant**: `http://localhost:8000/` (Arrival horizon planning, anti-herding dispatch, counterfactual recommendations).
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
- `POST /api/predict`: occupancy forecast for a zone and horizon (+15/+30/+60 min) with an 80% interval (`predicted_low/high`), `lag_source` (`observed|partial|estimated`) and `stream_frozen`.
- `GET /api/recommendations?destination=Block_26_CSE&horizon_minutes=15`: ranked parking grounds (destination matching is case/space tolerant); `POST /api/navigate` registers an in-flight driver dispatch.
- `GET /api/snapshot`: all-zone forecast with forecast ranges and server-side alerts; `GET /api/metrics`: benchmark table and classifier metrics.
- `GET /api/explain?zone_id=ZONE_BLOCK1&current_count=80`: SHAP breakdown of a forecast (base value + per-feature effect in cars); `GET /api/shap-figure/{name}`: generated diagnostic figures.
- `GET /api/demand-forecast?days=7&zone_id=ZONE_MAINPARK_NORTH&exam_dates=2026-11-23`: days-ahead peak time/level per zone with exam/event what-if.
- `GET /api/zone-analysis/{zone_id}` and `GET /api/heatmap?kind=zone_hour|zone_week|forecast`: usage profiles and heatmap matrices.
- `POST /api/ingest` and `GET /api/observations/stats`: collect real camera counts (optional `X-API-Key`, set `INGEST_API_KEY`).
- `POST /api/detect-image`, `POST /api/detect-frame`: YOLO detection (rate limited).

---

## 6. Data Sources, Serving Notes & Limitations

**Occupancy time series.** Simulated (see the caveat in Section 3). Replace it with real counts using the ingest endpoint and `python -m data_collection.store --out data/processed/lpu_live.csv`.

**Vision data.** The YOLO training images in `collab_content_backup/LPU---Parking-1` come from a Roboflow export (licence CC BY 4.0, see its `README.roboflow.txt`). The image file names follow the timestamp pattern of the public PKLot dataset, so confirm the original source and cite it before submission.

**Serving.** `/api/predict` builds lags from a rolling per-zone history (`models/history.py`, shared with `/api/ingest`) and reports `lag_source`; a frozen camera feed (6 identical readings over at least 10 minutes) sets `stream_frozen`. Counts in `/api/recommendations` and `/api/snapshot` are simulated demo values until the camera counter feeds the history buffer.

**Known limitations.** The test period contains no exam or event days (they fall in train/validation), so those regimes are not evaluated; intervals are calibrated on a single validation window; tuned parameters were searched at +15 min and reused for +30/+60. Demand-forecast exam/event effects rest on one exam week and two event days. The notebooks in `notebooks/01_exploratory_data_analysis.ipynb` and `02_model_training_and_tuning.ipynb` use their own simpler `GridSearchCV` run (MAE 4.9 at +15 min), which is weaker than the Optuna-tuned model used by the API (3.68).

**Not yet done:** real (logged) occupancy data actually collected, public deployment of the API, rolling-origin evaluation that holds out an exam week, drift monitoring.

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
