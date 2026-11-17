import os
import shutil
from pathlib import Path

src_root = Path(r"D:\Academic\Semester 5\ML\Supervised")
dst_root = Path(r"D:\Academic\Semester 5\ML\Campus-Parking-Occupancy-Prediction-Intelligent-Parking-Assistant")

items = [
    "data_pipeline/splitter.py",
    "features/feature_pipeline.py",
    "models/forecaster.py",
    "models/train.py",
    "models/inference.py",
    "evaluation/error_analysis.py",
    "explainability",
    "scripts",
    "notebooks/01_exploratory_data_analysis.ipynb",
    "notebooks/02_model_training_and_tuning.ipynb",
    "recommendation/parking_ranker.py",
    "computer_vision/detection.py",
    "api/limiter.py",
    "api/schemas.py",
    "api/routes.py",
    "api/main.py",
    "frontend/templates/analytics.html",
    "frontend/static/js/analytics.js",
    "frontend/templates/vision.html",
    "frontend/static/js/vision.js",
    "frontend/templates/zones.html",
    "frontend/static/js/zones.js",
    "frontend/static/js/driver.js",
    "frontend/static/js/admin.js",
    "tests/test_api.py",
    "artifacts/model_comparison.csv",
    "artifacts/pipelines",
    "artifacts/figures"
]

print("=== Syncing Supervised -> Git Repo ===")
for item in items:
    src_path = src_root / item
    dst_path = dst_root / item

    if not src_path.exists():
        print(f"Skipped (missing in src): {item}")
        continue

    if src_path.is_dir():
        dst_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        print(f"Synced Directory: {item}")
    else:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        print(f"Synced File:      {item}")

print("Sync completed successfully!")
