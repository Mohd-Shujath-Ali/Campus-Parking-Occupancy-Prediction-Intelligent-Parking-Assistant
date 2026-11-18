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
2. **Zero-Leakage Time-Series Pipeline**: Multi-horizon forecasting ($+15, +30, +60$ min) with a chronological 70/15/15 split (1 h buffer), 22 causal features: sin/cos time, zone identity, event/exam flags, lags ($t-15 \dots t-60$), velocity and shifted rolling statistics. Hyperparameters live in `configs/model_config.yaml`.
3. **Dual-Head Architecture**: Direct XGBoost Regressor per horizon paired with a Saturation Classifier ($P(\text{Occ} \ge 90\%)$) that handles class imbalance with `scale_pos_weight`.
4. **Inflection-Hour Evaluation**: On the simulated data, XGBoost cuts persistence MAE by roughly 59% inside $\pm 15$ minute class-transition windows (see Section 6 for caveats).
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

### 3. Run Unit & Integration Tests
```bash
pytest tests/ -v
```

---

## 4. REST API Documentation

- `GET /`: Serves the responsive web application dashboard.
- `GET /api/health`: System status, active zones count, forecast horizons.
- `GET /api/zones`: Full metadata for all 12 LPU parking grounds.
- `POST /api/predict`: Forecast occupancy for given zone, current count, and horizon (+15, +30, +60 min).
- `GET /api/recommendations?destination=Block_26_CSE&horizon_minutes=15`: Returns ranked viable parking grounds with walking distance, arrival availability, and smart tips.

---

## 5. Data Sources, Results & Limitations

**Occupancy time series (important).** `data/processed/lpu_timeseries.csv` is **simulated** by `data_pipeline/generator.py` (12 zones, 5-minute cadence, 56 days from 2026-09-01) from hand-written daily profiles per zone category, plus class-bell surges, an exam week, a campus event, rain days, a camera outage and Gaussian noise (σ = 2.5% of capacity). It is a stand-in until real detector counts are logged. Because the target is generated from these same profiles, scores below show that the pipeline works, **not** how well it would forecast real LPU traffic.

**Vision data.** The YOLO training images in `collab_content_backup/LPU---Parking-1` come from a Roboflow export (licence CC BY 4.0, see its `README.roboflow.txt`). The image file names follow the timestamp pattern of the public PKLot dataset, so confirm the original source and cite it before submission.

**Current results** (chronological hold-out, +15 min unless stated; simulated data):

| Model | MAE (cars) | RMSE | R² | MAE in bell windows |
| :--- | :---: | :---: | :---: | :---: |
| Persistence | 6.63 | 10.12 | 0.979 | 15.33 |
| Seasonal mean (zone × weekday × hour) | 6.35 | 9.93 | 0.980 | n/a |
| XGBoost +15 min | 4.21 | 5.69 | 0.994 | 6.28 |
| XGBoost +30 min | 4.49 | 6.08 | 0.993 | n/a |
| XGBoost +60 min | 4.96 | 6.84 | 0.991 | n/a |

The generator noise alone leaves an irreducible error of roughly 3-4 cars, so XGBoost is already close to the ceiling. The zone-aware seasonal mean is a strong baseline: the earlier 40-car MAE came from averaging all zones together.

**Serving.** `/api/predict` builds lags from a rolling per-zone history (`models/history.py`) and returns `lag_source` (`observed`, `partial` or `estimated`). Counts in `/api/recommendations` are simulated demo values (`count_source: simulated`) until the camera counter feeds the history buffer.

**Not yet done:** hyperparameter search / cross-validation, additional model comparisons (Ridge, Random Forest, stacking), SHAP explanations, error analysis and an EDA notebook.

---

## 6. Team Contribution (Rubric Component 10)

- **Mohammed Shujath Ali (12404599)**: Forecasting architecture, XGBoost forecaster, full-stack FastAPI + Tailwind web dashboard, temporal feature pipeline, project coordination.
- **SREEVYAS CS (12400848)**: Computer vision ROI polygon management, Ground contact point geometry, cross-camera Spatial NMS deduplication, FastAPI REST service, unit testing.
- **Dr. Mohit Arora (Faculty Supervisor)**: Project guidance and academic evaluation.
