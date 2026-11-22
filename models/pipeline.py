"""
scikit-learn Pipeline / ColumnTransformer used for every model.

One persisted object does preprocessing + prediction, so training and serving
share exactly the same transformations (no duplicated feature code):

  * zone_code           -> OneHotEncoder (zone is a category, not an ordinal)
  * count-like columns  -> StandardScaler (matters for Ridge, harmless for trees)
  * sin/cos, flags      -> passthrough (already bounded / binary)
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features.feature_pipeline import FEATURE_COLS, HORIZONS, horizon_col, zone_order

CATEGORICAL = ["zone_code"]
SCALED = [
    "vehicle_count", "operational_capacity", "occupancy_ratio",
    "lag_15m", "lag_30m", "lag_45m", "lag_60m", "delta_15m", "delta_30m",
    "rolling_mean_30m", "rolling_std_30m", "rolling_mean_60m", "rolling_std_60m",
    "profile_now", "profile_dev",
] + [horizon_col(h, f) for h in HORIZONS for f in ("profile", "mins_to_bell")]
PASSTHROUGH = [
    "hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend", "is_holiday",
    "is_exam_day", "is_campus_event", "event_load", "is_class_transition",
] + [horizon_col(h, f) for h in HORIZONS for f in ("event_load", "is_exam_day", "is_holiday")]
assert sorted(CATEGORICAL + SCALED + PASSTHROUGH) == sorted(FEATURE_COLS)


def build_preprocessor() -> ColumnTransformer:
    zone_codes = list(range(len(zone_order())))
    return ColumnTransformer(
        [
            ("zone", OneHotEncoder(categories=[zone_codes], handle_unknown="ignore",
                                   sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), SCALED),
            ("pass", "passthrough", PASSTHROUGH),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator) -> Pipeline:
    """Preprocessing + estimator as one persistable object."""
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def transformed_feature_names(pipe: Pipeline):
    """Readable column names after preprocessing (zone_code_3 -> zone name)."""
    order = zone_order()
    names = []
    for n in pipe.named_steps["prep"].get_feature_names_out():
        if n.startswith("zone_code_"):
            names.append("zone=" + order[int(n.rsplit("_", 1)[1])])
        else:
            names.append(n)
    return names
