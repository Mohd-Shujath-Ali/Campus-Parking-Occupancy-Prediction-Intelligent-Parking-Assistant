"""Shared data loading: chronological split -> profile (train only) -> features -> matrices."""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from data_pipeline.splitter import ChronologicalSplitter
from features.feature_pipeline import TimeSeriesFeaturePipeline
from features.profile import ProfileModel
from utils.config import load_model_config


@dataclass
class Split:
    X: pd.DataFrame
    y: Dict[str, pd.Series]
    meta: pd.DataFrame  # timestamp, zone_id (row-aligned with X)


def split_raw(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = load_model_config()["data"]
    return ChronologicalSplitter(d["train_ratio"], d["val_ratio"], d["test_ratio"],
                                 d["chronological_buffer_hours"]).split_dataframe(df)


def build_splits(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: Optional[pd.DataFrame] = None,
                 profile: Optional[ProfileModel] = None) -> Tuple[Dict[str, Split], ProfileModel]:
    """
    The profile is fitted on the TRAIN period only. Train rows see a leave-one-week-out profile (their
    own week excluded); validation, test and live rows see the full train profile.
    """
    f = load_model_config()["features"]
    profile = profile or ProfileModel().fit(df_train)
    kw = dict(lag_minutes=f["lag_minutes"], rolling_windows_minutes=f["rolling_windows_minutes"],
              saturation_threshold=f["saturation_threshold"], profile=profile)
    train_pipe = TimeSeriesFeaturePipeline(profile_leave_week_out=True, **kw)
    eval_pipe = TimeSeriesFeaturePipeline(**kw)
    out = {}
    for name, part, pipe in (("train", df_train, train_pipe), ("val", df_val, eval_pipe), ("test", df_test, eval_pipe)):
        if part is None:
            continue
        X, y, meta = pipe.prepare_with_meta(pipe.transform_zone_series(part))
        out[name] = Split(X, y, meta)
    return out, profile


def load_dataset(csv_path: str = "data/processed/lpu_timeseries.csv"):
    """(splits, profile, raw frames) for the standard train / validation / holdout-test protocol."""
    df = pd.read_csv(csv_path)
    parts = split_raw(df)
    splits, profile = build_splits(*parts)
    return splits, profile, dict(zip(("train", "val", "test"), parts))


def load_splits(csv_path: str = "data/processed/lpu_timeseries.csv") -> Dict[str, Split]:
    """Split FIRST (chronologically, with buffer), then build features per split."""
    return load_dataset(csv_path)[0]


def cv_gap_rows(n_zones: int = 12) -> int:
    """Rows to skip between CV train/validation folds (cv_gap steps x zones)."""
    return load_model_config()["training"]["cv_gap"] * n_zones
