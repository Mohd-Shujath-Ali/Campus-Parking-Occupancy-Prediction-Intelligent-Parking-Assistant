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
1. **Perception**: Fine-tuned YOLO (`best.pt`, mAP50=91.6%) and Pretrained YOLO (`yolo26l.pt`) with Ground-Contact Point Jordan Ray-Casting Point-in-Polygon (PIP), cross-camera Global Spatial NMS ($d < 2.5\text{m}$), and OpenCV pre-inference quality diagnostics (blur, underexposure, glare).
2. **Zero-Leakage Time-Series Pipeline**: Multi-horizon forecasting ($+15, +30, +60$ min) combining continuous $\sin/\cos$ harmonic time, causal lags ($t-15, t-30, t-45, t-60$), shifted rolling demand velocity, and academic bell surge indicators, isolated across chronological partitions (70% Train, 15% Val, 15% Holdout Test with a 1-hour purge buffer).
3. **Multi-Model Comparative Benchmarking**: Persistence and Seasonal Historical Mean baselines, Linear $L_2$ Ridge Regression, Non-Linear Random Forest Ensembles, and `TimeSeriesSplit(n_splits=5, gap=12)` tuned XGBoost Regressors.
4. **Dual-Head Architecture**: Multi-horizon continuous count regressor paired with a balanced Saturation Classifier ($P(\text{Occ} \ge 90\%)$) evaluated via Precision-Recall AUC (PR-AUC = 0.9498).
5. **Explainable AI (SHAP)**: Global feature beeswarm attribution, mean $|SHAP|$ bar rankings, and local waterfall decision traces for rush-hour transitions and steady states.
6. **Error Analysis & Diagnostics**: Residuals vs. Fitted plots, normal distribution Q-Q plots, and cross-zone spatial error breakdown.
7. **Anti-Herding Traffic Dispersion**: Real-time `InFlightDispatchTracker` with 15-minute TTL buffering and Softmax traffic dispersion to eliminate herd collapse.
8. **Full-Stack Web Microservice**: High-performance asynchronous **FastAPI** backend with SlowAPI rate limiting, serving responsive multi-page **TailwindCSS & Chart.js** interfaces.

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

| Model Architecture | Forecast Horizon | Data Partition | MAE (Cars) | RMSE (Cars) | $R^2$ Score | Surge Inflection MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | +15 min | Holdout Test | 6.064 | 9.209 | 0.9805 | 15.571 |
| **Seasonal Historical Mean** | +15 min | Holdout Test | 40.399 | 53.852 | 0.3339 | — |
| **Ridge Regression ($L_2$)** | +15 min | Holdout Test | 5.607 | 8.350 | 0.9840 | 15.846 |
| **Random Forest Ensemble** | +15 min | Holdout Test | 4.069 | 5.539 | 0.9930 | 7.296 |
| **Tuned XGBoost** | +15 min | Validation | 4.634 | 6.354 | 0.9906 | — |
| **Tuned XGBoost** | +15 min | Holdout Test | **4.660** | **6.442** | **0.9905** | **9.839** (~40% vs Pers.) |
| **Tuned XGBoost** | +30 min | Holdout Test | 5.360 | 7.451 | 0.9872 | — |
| **Tuned XGBoost** | +60 min | Holdout Test | 6.743 | 9.739 | 0.9782 | — |
| **Saturation Classifier** | +15 min | Holdout Test | — | — | — | **PR-AUC = 0.9498** (Recall: 98.7%) |

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
*Outputs: Evaluates all candidate baselines, runs 5-fold TimeSeriesSplit GridSearchCV, trains multi-horizon estimators, generates error analysis plots (`residuals_vs_fitted.png`, `error_distribution_qq.png`, `error_by_zone_boxplot.png`), and serializes model wrappers to `artifacts/pipelines/`.*

### 2. Generate Explainable AI (SHAP) Interpretability Suite
```bash
python -m explainability.shap_analysis
```
*Outputs: Generates TreeSHAP summary beeswarm, feature importance bar charts, and local rush-hour / steady-state decision waterfalls to `artifacts/figures/`.*

### 3. Launch FastAPI Web Application & API Server
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Driver Assistant**: `http://localhost:8000/` (Arrival horizon planning, anti-herding dispatch, counterfactual recommendations).
- **Admin Command Center**: `http://localhost:8000/admin` (Real-time occupancy charts, camera sensor telemetry, invariant stream detection).
- **Vision & ROI Studio**: `http://localhost:8000/vision` (Live webcam stream, interactive polygon/box ROI drawing, dual-model YOLO benchmarking).
- **Model Analytics & XAI**: `http://localhost:8000/analytics` (Benchmark comparisons, SHAP attribution tabs, residual diagnostics).
- **Campus Topology & Heatmap**: `http://localhost:8000/zones` (Live 12-zone congestion heatmap grid and grounds directory).
- **Interactive REST Swagger**: `http://localhost:8000/docs`.

### 4. Run Test Suite
```bash
pytest tests/ -v
```
*Runs 26 automated unit and integration tests verifying API endpoints, data pipelines, anti-herding dispatch, and computer vision filters.*

---

## 5. Jupyter Research Notebooks

The `notebooks/` directory contains structured, reproducible experiments aligned with the course rubrics:
- `notebooks/01_exploratory_data_analysis.ipynb`: Data quality checks, ADF stationarity tests, ACF/PACF lag selection, bell surge distributions, exam period density shifts, saturation class imbalance, and correlation heatmaps.
- `notebooks/02_model_training_and_tuning.ipynb`: Chronological train/val/test splitting, baseline vs. ML model comparisons, TimeSeriesSplit GridSearchCV curves, surge inflection error breakdown, and dual-head classification evaluation.
- `notebooks/smote_experiment.ipynb`: Oversampling and imbalance handling experiments.

---

## 6. Team Contributions

- **Mohammed Shujath Ali (12404599)**: Forecasting architecture, TimeSeriesSplit hyperparameter tuning, TreeSHAP interpretability, full-stack FastAPI microservice, TailwindCSS/Chart.js frontends, anti-herding dispatch tracker, and project coordination.
- **SREEVYAS CS (12400848)**: Computer vision ROI polygon management, Ground contact point geometry, cross-camera Spatial NMS deduplication, image degradation diagnostics, test suite development.
- **Dr. Mohit Arora (Faculty Supervisor)**: Project mentorship, technical guidance, and academic evaluation.
