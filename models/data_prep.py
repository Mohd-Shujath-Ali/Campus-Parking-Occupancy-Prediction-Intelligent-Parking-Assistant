"""Shared data loading for experiments: chronological split -> features -> matrices."""
from dataclasses import dataclass
from typing import Dict

import pandas as pd

from data_pipeline.splitter import ChronologicalSplitter
from features.feature_pipeline import TimeSeriesFeaturePipeline
from utils.config import load_model_config


@dataclass
class Split:
    X: pd.DataFrame
    y: Dict[str, pd.Series]
    meta: pd.DataFrame  # timestamp, zone_id (row-aligned with X)


def load_splits(csv_path: str = "data/processed/lpu_timeseries.csv") -> Dict[str, Split]:
    """Split FIRST (chronologically, with buffer), then build features per split."""
    cfg = load_model_config()
    d, f = cfg["data"], cfg["features"]
    df = pd.read_csv(csv_path)
    splitter = ChronologicalSplitter(d["train_ratio"], d["val_ratio"], d["test_ratio"],
                                     d["chronological_buffer_hours"])
    parts = dict(zip(("train", "val", "test"), splitter.split_dataframe(df)))
    pipe = TimeSeriesFeaturePipeline(f["lag_minutes"], f["rolling_windows_minutes"],
                                     saturation_threshold=f["saturation_threshold"])
    out = {}
    for name, part in parts.items():
        X, y, meta = pipe.prepare_with_meta(pipe.transform_zone_series(part))
        out[name] = Split(X, y, meta)
    return out


def cv_gap_rows(n_zones: int = 12) -> int:
    """Rows to skip between CV train/validation folds (cv_gap steps x zones)."""
    return load_model_config()["training"]["cv_gap"] * n_zones
