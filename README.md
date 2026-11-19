# Campus Parking Occupancy Prediction & Intelligent Parking Assistant (Project 32)
## School of Computing & Artificial Intelligence — Lovely Professional University (LPU)

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-v8%20%2F%20v26-orange.svg)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.115-009688.svg)](https://fastapi.tiangolo.com)
[![HTML5/Tailwind](https://img.shields.io/badge/Frontend-HTML5%20%7C%20TailwindCSS%20%7C%20Chart.js-38B2AC.svg)](https://tailwindcss.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.1-red.svg)](https://xgboost.readthedocs.io/)

---

## 1. Project Overview

This project delivers an end-to-end Machine Learning, Computer Vision, and Decision Support system for parking management across the 600-acre Lovely Professional University (LPU) campus in Phagwara, Punjab.

Unlike traditional marked-stall sensor systems, this architecture addresses **open, unstructured campus grounds** using:
1. **Perception**: Fine-tuned YOLO (`best.pt`) with Ground-Contact Point Jordan Ray-Casting Point-in-Polygon (PIP) and cross-camera Global Spatial NMS ($d < 2.5\text{m}$).
2. **Zero-Leakage Time-Series Pipeline**: Multi-horizon forecasting ($+15, +30, +60$ min) with a chronological 70/15/15 split (1 h buffer), 22 causal features: sin/cos time, zone identity, event/exam flags, lags ($t-15 \dots t-60$), velocity and shifted rolling statistics. Hyperparameters live in `configs/model_config.yaml` (Optuna-tuned overrides in `configs/tuned_params.yaml`).
3. **Dual-Head Architecture**: Direct XGBoost Regressor per horizon paired with a Saturation Classifier ($P(\text{Occ} \ge 90\%)$) that handles class imbalance with `scale_pos_weight` and is isotonic-calibrated on the validation period.
4. **Inflection-Hour Evaluation**: On the simulated data, XGBoost cuts persistence MAE by roughly 67% inside $\pm 15$ minute class-transition windows (see Section 6 for caveats).
5. **Anti-Herding Dispatcher**: Recommendation ranker that picks the lead lot among near-tied zones with softmax($\text{score}/\beta$) probabilities, so drivers are spread across equivalent lots.
6. **Full-Stack Web Microservice**: High-performance asynchronous **FastAPI** backend with unified responsive **TailwindCSS & Chart.js** web interface (no Streamlit).

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

## 3. Quick Start & Execution Guide

### Prerequisites & Installation
```bash
# Clone the repository
git clone https://github.com/vibin-123/Campus-Parking-Occupancy-Prediction-Intelligent-Parking-Assistant.git
cd Campus-Parking-Occupancy-Prediction-Intelligent-Parking-Assistant

# Install pinned dependencies
pip install -r requirements.txt
```

### 1. Run End-to-End Training & Baseline Benchmarking
```bash
python -m models.train
```
*Outputs: loads `data/processed/lpu_timeseries.csv` (regenerate it with `python -m data_pipeline.generator`), fits Persistence, zone-aware Seasonal Mean and XGBoost models, and exports `artifacts/model_comparison.csv` plus serialised `.joblib`/`.ubj` models.*

### 2. Launch FastAPI Web Application & API Server

#### Option A: Run via Docker (Recommended)
```bash
# Start all containerized services with Docker Compose
docker compose up --build -d

# Check container status & health
docker compose ps
docker compose logs -f
```
See [DOCKER.md](DOCKER.md) for full containerization details and GPU configuration.

#### Option B: Run Locally
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Web App**: Open `http://localhost:8000/` in your browser for the responsive Driver Assistant & Campus Estate Console.
- **REST API Docs**: Open `http://localhost:8000/docs` for interactive Swagger documentation.

### 3. Reproduce the experiments
```bash
python -m experiments.compare_models   # Ridge / RF / XGBoost / LightGBM / stacking vs baselines
python -m experiments.tune_xgboost     # Optuna TPE vs random search, TimeSeriesSplit CV -> configs/tuned_params.yaml
python -m models.train                 # fits + saves the deployed sklearn Pipelines (uses tuned params)
python -m experiments.error_analysis   # per-zone / hour / regime errors, worst cases, classifier calibration
python -m experiments.explain          # SHAP global + local explanations
```

### 4. Run Unit & Integration Tests
```bash
pytest tests/ -v
```

---

## 4. REST API Documentation

- `GET /`: Serves the responsive web application dashboard.
- `GET /api/health`: System status, active zones count, forecast horizons.
- `GET /api/zones`: Full metadata for all 12 LPU parking grounds.
- `GET /api/explain?zone_id=ZONE_BLOCK1&current_count=80`: SHAP breakdown of a forecast (base value + per-feature effect in cars).
- `GET /api/demand-forecast?days=7&zone_id=ZONE_MAINPARK_NORTH&exam_dates=2026-11-23`: days-ahead peak time/level per zone with exam/event what-if.
- `GET /api/zone-analysis/{zone_id}` and `GET /api/heatmap?kind=zone_hour|zone_week|forecast`: usage profiles and heatmap matrices.
- `POST /api/ingest` and `GET /api/observations/stats`: collect real camera counts (optional `X-API-Key`).
- `POST /api/predict`: Forecast occupancy for given zone, current count, and horizon (+15, +30, +60 min).
- `GET /api/recommendations?destination=Block_26_CSE&horizon_minutes=15`: Returns ranked viable parking grounds with walking distance, arrival availability, and smart tips.

---

## 5. Data Sources, Results & Limitations

**Occupancy time series (important).** `data/processed/lpu_timeseries.csv` is **simulated** by `data_pipeline/generator.py` (12 zones, 5-minute cadence, 56 days from 2026-09-01) from hand-written daily profiles per zone category, plus class-bell surges, an exam week, a campus event, rain days, a camera outage and Gaussian noise (σ = 2.5% of capacity). It is a stand-in until real detector counts are logged. Because the target is generated from these same profiles, scores below show that the pipeline works, **not** how well it would forecast real LPU traffic.

**Vision data.** The YOLO training images in `collab_content_backup/LPU---Parking-1` come from a Roboflow export (licence CC BY 4.0, see its `README.roboflow.txt`). The image file names follow the timestamp pattern of the public PKLot dataset, so confirm the original source and cite it before submission.

**Current results** (chronological hold-out, +15 min unless stated; simulated data; full write-up in [`docs/results.md`](docs/results.md)):

| Model | MAE (cars) | RMSE | R² | MAE in bell windows |
| :--- | :---: | :---: | :---: | :---: |
| Persistence | 6.63 | 10.12 | 0.979 | 15.33 |
| Seasonal mean (zone x weekday x hour) | 6.35 | 9.93 | 0.980 | 12.95 |
| Ridge | 6.13 | 9.16 | 0.983 | 15.29 |
| Random Forest | 4.16 | 5.68 | 0.994 | 7.34 |
| LightGBM | 4.41 | 6.01 | 0.993 | 6.89 |
| XGBoost (config defaults) | 4.21 | 5.71 | 0.993 | 6.32 |
| Stacking (RF + tuned XGB + LGBM) | 3.94 | 5.34 | 0.994 | 6.03 |
| **XGBoost (Optuna-tuned)** | **3.68** | **4.95** | **0.995** | **5.05** |

Tuned XGBoost: 3.72 / 3.79 MAE at +30 / +60 min (persistence 9.46 / 14.84). Bayesian search beat random search with the same 30-trial budget (CV MAE 4.43 vs 4.50). The best a forecaster could do on this simulator (noise-free oracle) is 3.38 cars, so the model is within ~0.3 cars of the ceiling and the data, not the model, is now the limiting factor. The saturation classifier is isotonic-calibrated on the validation period (precision 0.87, recall 0.75, PR-AUC 0.90 at threshold 0.5).

**Explainability.** SHAP (global, dependence and local plots in `artifacts/figures/shap_*.png`, interpretation in `docs/results.md`); live per-prediction breakdown at `GET /api/explain`.

**EDA.** [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) (data quality, distributions, daily/weekly patterns, special regimes, autocorrelation, class imbalance, split composition).

**Serving.** `/api/predict` builds lags from a rolling per-zone history (`models/history.py`) and returns `lag_source` (`observed`, `partial` or `estimated`). Counts in `/api/recommendations` are simulated demo values (`count_source: simulated`) until the camera counter feeds the history buffer.

**Also implemented:** 80% conformal prediction intervals (test coverage 0.81-0.82), calendar-based demand forecasting with exam/event what-if (occupancy MAE 2.7% of capacity vs 3.4% seasonal mean), zone-level analysis, heatmaps, admin alerts and a camera-count ingest + export pipeline (see `docs/results.md` section 7).

**Not yet done:** real (logged) occupancy data actually collected, public deployment, rolling-origin evaluation that holds out an exam week, drift monitoring.

---

## 6. Team Contribution (Rubric Component 10)

- **Mohammed Shujath Ali (12404599)**: Forecasting architecture, XGBoost forecaster, full-stack FastAPI + Tailwind web dashboard, temporal feature pipeline, project coordination.
- **SREEVYAS CS (12400848)**: Computer vision ROI polygon management, Ground contact point geometry, cross-camera Spatial NMS deduplication, FastAPI REST service, unit testing.
- **Dr. Mohit Arora (Faculty Supervisor)**: Project guidance and academic evaluation.
