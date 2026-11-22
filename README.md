# Campus Parking Occupancy Prediction & Intelligent Parking Assistant (Project 32)
## School of Computer Science & Engineering — Lovely Professional University (LPU)

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-ali1131%2Fcampus--parking-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/ali1131/campus-parking)
[![Render Deployment](https://img.shields.io/badge/Render-Docker%20Live-46E3B7?logo=render&logoColor=white)](https://render.com)
[![Ultralytics](https://img.shields.io/badge/YOLO-v8%20%7C%20v26%20ONNX-orange.svg)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.115-009688.svg)](https://fastapi.tiangolo.com)
[![HTML5/Tailwind](https://img.shields.io/badge/Frontend-TailwindCSS%20%7C%20Chart.js-38B2AC.svg)](https://tailwindcss.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.1-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-TreeSHAP-brightgreen.svg)](https://shap.readthedocs.io/)

---

## 1. Project Overview

This project delivers an end-to-end Machine Learning, Computer Vision, and Decision Support system for parking management across the 600-acre Lovely Professional University (LPU) campus in Phagwara, Punjab.

Unlike traditional marked-stall sensor systems, this architecture addresses **open, unstructured campus grounds** using:
1. **Perception**: Fine-tuned YOLO (`best.pt`, mAP50=91.6%) and Pretrained YOLO (`yolo26l.pt`), with an in-browser ONNX Runtime Web model (`yolo26l.onnx`) running via WebGPU/WASM for zero-server-cost live camera inference. Features Ground-Contact Point Jordan Ray-Casting Point-in-Polygon (PIP), cross-camera Global Spatial NMS ($d < 2.5\text{m}$), and OpenCV pre-inference quality diagnostics (blur, underexposure, glare).
2. **Zero-Leakage Time-Series Pipeline**: Multi-horizon forecasting ($+15, +30, +60$ min, interpolated to any arrival time) with a chronological 70/15/15 split (1 h purge buffer) and 41 causal features: sin/cos time, zone identity, lags ($t-15 \dots t-60$), velocity, shifted rolling statistics, a leave-one-week-out **normal-day profile** per zone and time of week, and what is already known about the *arrival* time (events, exams, holidays, distance to a lecture bell). Calendar data comes from the admin-managed campus calendar. All preprocessing lives in one persisted scikit-learn `Pipeline`/`ColumnTransformer`; hyperparameters are in `configs/model_config.yaml` (Optuna-tuned overrides in `configs/tuned_params.yaml`). Design, evidence and limits: [`docs/FORECASTING.md`](docs/FORECASTING.md).
3. **Multi-Model Comparative Benchmarking**: Persistence, normal-day-profile and zone-aware Seasonal Mean baselines, Ridge, Random Forest (bagging) and XGBoost (boosting), compared with `experiments/forecast_ablation.py`; boosting wins and a bagging blend adds nothing. Hyperparameters were Bayesian-tuned (Optuna TPE) on the earlier dataset with `TimeSeriesSplit` cross-validation; an optional `GridSearchCV` runner is kept in `models/train.py --grid-search`.
4. **Dual-Head Architecture**: Multi-horizon count regressor plus a Saturation Classifier ($P(\text{Occ} \ge 90\%)$) using `scale_pos_weight` and isotonic probability calibration on the validation period.
5. **Uncertainty & Demand Forecasting**: Calibrated 80% ranges (held-out coverage 0.79-0.80, 0.82-0.89 during events), the **chance of a free slot on arrival** (Brier 0.012-0.016 vs 0.056 for the base rate), and a calendar-only demand forecaster for days-ahead planning that applies the campus calendar automatically.
6. **Explainable AI (SHAP)**: Global beeswarm and mean $|SHAP|$ rankings, dependence plots, local waterfall traces, and a live per-prediction breakdown at `GET /api/explain`.
7. **Error Analysis & Diagnostics**: Residuals vs. fitted, Q-Q plots, per-zone / per-hour / per-regime error tables, worst-case review and classifier calibration.
8. **Anti-Herding Traffic Dispersion**: A softmax lead-zone selector among near-tied lots plus a real-time `InFlightDispatchTracker` (15-minute TTL, 35% capacity cap) fed by `POST /api/navigate`.
9. **Data Collection & Ingestion**: SQLite store of per-camera counts, summed per zone, with `POST /api/ingest` and an exporter that writes real observations in the training schema.
10. **Full-Stack Containerized Microservice**: Asynchronous **FastAPI** backend with SlowAPI rate limiting, serving responsive multi-page **TailwindCSS & Chart.js** interfaces (admin alerts, heatmaps, 7-day demand outlook, zone usage profiles). Containerized with **Docker**, published to **Docker Hub** (`ali1131/campus-parking`), and deployed as a live cloud service on **Render**.
11. **Presentation Mode (Zero-Hustle Demo)**: A 1-click auto-login persona suite directly on the `/login` screen for instant live presentation across Student, Faculty, Warden, Guard, Vendor, and Admin roles without manual credential typing.

---

## 2. Presentation & Live Demo Guide (1-Click Auto-Login)

For academic evaluations, presentations, and viva demonstrations, the login screen (`/login`) includes **1-Click Presentation Personas**. Clicking any persona instantly autofills credentials and signs in directly to their tailored view.

All presentation demo accounts share the universal demo password: **`CampusPass@2026`** (no forced password change on first login, active and unlocked out of the box).

| Persona | Primary ID | Quick Alias | Role / Department | Landing View & Capabilities |
| :--- | :--- | :--- | :--- | :--- |
| 🎓 **Student** | `12301234` | `student` | Aarav Sharma · School of CSE | Auto-redirects to `/` (Driver View) personalized for **Block 26 (School of CSE)** with live walking distances |
| 👨‍🏫 **Faculty** | `26541` | `faculty` | Dr. Meera Kapoor · Admin / Prof | Auto-redirects to `/` (Driver View) personalized for **Block 1 (Central Administration)** |
| 🏢 **Hostel Warden** | `W-BH2-01` | `warden` | Rajinder Singh · Boys Hostel 2 | Auto-redirects to `/admin` (Estate Console) with hostel zone overrides (`FULL`, `CLOSED`) and violation reports |
| 👮 **Security Guard** | `G-GATE-03` | `guard` | Harpreet Kaur · Gate Security | Auto-redirects to `/admin` (Estate Console) for live gate telemetry, incident logging, and violations |
| ⚡ **Administrator** | `admin` | `admin` | Chief Parking Administrator | Auto-redirects to `/admin` with **full master access** across all 8 modules (Cameras & ROI, YOLO Vision, Analytics, Users, Calendar) |
| 🚚 **Vendor** | `V-CANTEEN-07` | `vendor` | Punjab Caterers · MB Food Square | Auto-redirects to `/` (Driver View) with delivery bay parking access |

> **Presentation Tip:** You can also manually type either the primary ID (e.g. `12301234`) or the short alias (e.g. `student`) with password `CampusPass@2026`. The login rate limit is configured to 60 requests/minute to allow rapid persona switching during live demonstrations.

---

## 3. Monitored LPU Campus Zones (Empirical Satellite Survey)

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

## 4. Machine Learning Benchmark Results (Holdout Test Split)

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

Raw tables: `artifacts/model_comparison.csv`, `artifacts/regime_metrics.csv`, `artifacts/ablation.csv`, `artifacts/recommendation_eval.csv`.

---

## 5. Quick Start & Execution Guide

### Local Installation
```bash
# 1. Clone the repository
git clone https://github.com/Mohd-Shujath-Ali/Campus-Parking-Occupancy-Prediction-Intelligent-Parking-Assistant.git
cd Campus-Parking-Occupancy-Prediction-Intelligent-Parking-Assistant

# 2. Install dependencies
pip install -r requirements.txt       # Lean runtime (FastAPI, dashboards, ML inference)
# pip install -r requirements-full.txt  # Full dev environment (model retraining, SHAP, tests)
```

### Launch FastAPI Server Locally
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000/login`** in your browser. Everyone signs in at `/login` with 1-click presentation personas or individual credentials:
- **Driver Assistant**: `http://localhost:8000/` (arrival-time slider, chance of a free slot per ground, walking distance ranking, computed arrival tips, anti-herding dispatch).
- **Campus Calendar** (admin): `http://localhost:8000/calendar` (events with venue and expected vehicles, exam windows, holidays; forecasts use them immediately).
- **Model Health** (admin): `http://localhost:8000/model` (model version and training data, live accuracy and drift).
- **Admin Command Center**: `http://localhost:8000/admin` (Real-time occupancy charts with 80% forecast ranges, operational alerts, camera sensor telemetry, invariant stream detection).
- **Vision & ROI Studio**: `http://localhost:8000/vision` (Live webcam stream, interactive polygon/box ROI drawing, in-browser ONNX YOLO benchmarking).
- **Model Analytics & XAI**: `http://localhost:8000/analytics` (Benchmark comparisons, SHAP attribution tabs, residual diagnostics, zone × hour demand heatmaps, 7-day demand outlook).
- **Campus Topology & Heatmap**: `http://localhost:8000/zones` (Live 12-zone congestion heatmap grid, grounds directory and per-zone usage profiles).
- **Interactive REST Swagger**: `http://localhost:8000/docs`.

---

## 6. Docker Containerization & Cloud Deployment (Docker Hub & Render)

The entire system is containerized with Docker, optimized for low memory footprints (~310 MB RAM resident), and deployed as a live cloud microservice.

### A. Docker Image Build & Local Run
Build using the optimized Linux x86_64 target:
```bash
# Build the production Docker image
docker build --platform linux/amd64 -f Dockerfile.render -t ali1131/campus-parking:latest .

# Run the container locally
docker run -d -p 8000:10000 --name campus-parking-app ali1131/campus-parking:latest
```
Access the application at `http://localhost:8000`.

### B. Push to Docker Hub
```bash
# Push image to Docker Hub
docker push ali1131/campus-parking:latest
```
Image repository: **[hub.docker.com/r/ali1131/campus-parking](https://hub.docker.com/r/ali1131/campus-parking)**

### C. Deploy on Render
1. In the **Render Dashboard**, select **New +** $\rightarrow$ **Web Service**.
2. Choose **Existing Image** (Docker Hub) and input: `ali1131/campus-parking:latest`.
3. Set the Health Check path to `/api/health`.
4. (Optional) Set environment variables:
   - `DEMO_LOGINS=on` (Enables presentation demo cards)
   - `LOGIN_RATE_LIMIT=60/minute`
5. Click **Deploy Web Service**. Render pulls directly from Docker Hub and starts the service without build timeouts!

---

## 7. Model Training, Interpretability & Diagnostics

### 1. Run End-to-End Training
```bash
python -m models.train
```
*Outputs: loads `data/processed/lpu_timeseries.csv`, fits baselines, Ridge, Random Forest, tuned XGBoost (multi-horizon), the calibrated saturation classifier, conformal intervals, and demand forecaster. Serializes pipelines to `artifacts/pipelines/`.*

### 2. Generate Explainable AI (TreeSHAP) Suite
```bash
python -m explainability.shap_analysis
```
*Outputs: Generates TreeSHAP summary beeswarm plots, feature importance bar charts, and local rush-hour / steady-state decision waterfalls to `artifacts/figures/`.*

### 3. Reproduce Experimental Ablations
```bash
python -m experiments.forecast_ablation              # old vs new features, direct vs profile-relative, bagging vs boosting
python -m experiments.interval_comparison            # CQR vs residual-distribution intervals
python -m experiments.replay_recommendations --tune  # choose ranker weights on the validation period
python -m experiments.replay_recommendations         # slot finding rate on holdout test set
```

### 4. Run Automated Test Suite
```bash
pytest tests/ -v
```
*Executes unit and integration tests (REST API, time-series feature pipeline, prediction intervals, anti-herding dispatch, stream-freeze diagnostics, computer-vision polygon raycasting).*

---

## 8. REST API Endpoints

- `GET /api/health`, `GET /api/zones`: System status and metadata for all 12 zones.
- `POST /api/auth/login`: Session authentication with scrypt hashing, returning role landing target.
- `POST /api/predict`: Multi-horizon occupancy forecast (5–60 min) with 80% interval (`predicted_low/high`), `p_free`, and `lag_source`.
- `GET /api/recommendations?destination=Block_26_CSE&horizon_minutes=30`: Distance-ranked grounds with `p_free`, capacity, and walking estimates.
- `POST /api/navigate`: Registers an in-flight driver dispatch to disperse traffic.
- `GET /api/snapshot`: Real-time camera count/estimate for every zone, +15 min forecast, and operational warnings.
- `GET /api/explain?zone_id=ZONE_BLOCK1&current_count=80`: Dynamic SHAP breakdown of a live forecast (base value + per-feature effect in cars).
- `GET /api/demand-forecast?days=7&zone_id=ZONE_MAINPARK_NORTH`: 7-day peak occupancy planner with exam/event what-if parameters.
- `GET /api/heatmap?kind=zone_hour|zone_week|forecast`: Congestion heatmap matrices.
- `POST /api/ingest`: Ingestion endpoint for camera counts with optional `X-API-Key`.
- `POST /api/detect-image`: YOLO detection with quality diagnostics.

---

## 9. Data Sources & Serving Notes

- **Occupancy Time Series:** Simulated baseline dataset (12 zones, 5-minute cadence, 18 weeks). Real counts replace it dynamically through `POST /api/ingest`. `python -m scripts.simulate_ingest` feeds real-time simulated counts for testing.
- **Vision Data & In-Browser Inference:** Pretrained YOLOv8 / YOLO26l models converted to ONNX format. The client-side browser inference (`client_yolo.js`) executes directly via WebAssembly/WebGPU, eliminating server GPU bottlenecks.
- **Serving Architecture:** `/api/recommendations` and `/api/snapshot` query rolling observation history rehydrated into memory at startup. Live streams, stale fallbacks, and frozen stream detection are monitored automatically.
- **Active Progress:** The application is fully containerized and hosted on Render via Docker Hub (`ali1131/campus-parking:latest`). Ingestion pipelines are ready for live RTSP campus camera feeds.

---

## 10. Jupyter Research Notebooks

The `notebooks/` directory contains structured, reproducible experiments aligned with the course rubrics:
- `notebooks/01_exploratory_data_analysis.ipynb`: Data quality checks, ADF stationarity tests, ACF/PACF lag selection, bell surge distributions, exam period density shifts, saturation class imbalance, and correlation heatmaps.
- `notebooks/02_model_training_and_tuning.ipynb`: Chronological train/val/test splitting, baseline vs. ML model comparisons, TimeSeriesSplit GridSearchCV curves, surge inflection error breakdown, and dual-head classification evaluation.
- `notebooks/01_eda.ipynb`: Independent exploratory analysis of seasonal regimes, special events, and class distributions.
- `notebooks/smote_experiment.ipynb`: Oversampling and imbalance handling experiments.

---

## 11. Team Contributions

- **Mohammed Shujath Ali (12404599)**: Forecasting architecture, TimeSeriesSplit hyperparameter tuning, TreeSHAP interpretability, full-stack FastAPI microservice, TailwindCSS/Chart.js frontends, Dockerization & Render cloud deployment, 1-click presentation authentication suite, and project coordination.
- **SREEVYAS CS (12400848)**: Computer vision ROI polygon management, Ground contact point geometry, cross-camera Spatial NMS deduplication, image degradation diagnostics, test suite development.
- **Dr. Mohit Arora (Faculty Supervisor)**: Project mentorship, technical guidance, and academic evaluation.
