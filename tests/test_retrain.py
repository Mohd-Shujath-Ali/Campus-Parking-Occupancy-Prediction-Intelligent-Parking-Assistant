"""Retraining on real counts: readiness checks, timeline assembly, champion/challenger gating."""
import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from data_collection.store import ObservationStore
from data_pipeline.generator import generate_lpu_parking_dataset
from models import retrain
from models.retrain import assemble, retrain as run_retrain

TINY = {"n_estimators": 25, "max_depth": 5}


def _store_with(df: pd.DataFrame) -> ObservationStore:
    """An observation store holding `df` (generator rows) as if cameras had reported it."""
    st = ObservationStore(":memory:")
    rows = [(r.timestamp.isoformat(), r.zone_id, f"CAM_{r.zone_id}", int(r.vehicle_count), "camera", "2026-01-01T00:00:00")
            for r in df.itertuples()]
    with st._conn() as c:
        c.executemany("INSERT INTO observations VALUES (?,?,?,?,?,?)", rows)
    return st


@pytest.fixture(scope="module")
def synth_csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("synth") / "synthetic.csv"
    generate_lpu_parking_dataset(start_date="2026-08-03", days=21, write=False).to_csv(p, index=False)
    return str(p)


@pytest.fixture(scope="module")
def real_df():
    df = generate_lpu_parking_dataset(start_date="2026-12-07", days=28, write=False, seed=7)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_not_enough_real_data_keeps_the_current_model(tmp_path):
    out = run_retrain(ObservationStore(":memory:"), out_dir=tmp_path / "p", log=lambda *_: None)
    assert out["decision"] == "insufficient_data"
    one_day = generate_lpu_parking_dataset(start_date="2026-12-07", days=2, write=False)
    out = run_retrain(_store_with(one_day), out_dir=tmp_path / "p", log=lambda *_: None)
    assert out["decision"] == "insufficient_data" and not (tmp_path / "p").exists()


def test_assemble_puts_real_data_last_with_weekdays_aligned(real_df, synth_csv):
    synth = pd.read_csv(synth_csv)
    real = real_df.drop(columns=["camera_status", "camera_age_seconds"]).copy()
    a = assemble(real, synth)
    df = a["df"]
    assert a["mode"] == "mixed"
    s_part, r_part = df[df.sample_weight == 1.0], df[df.sample_weight == retrain.REAL_WEIGHT]
    assert s_part.timestamp.max() < r_part.timestamp.min()                         # synthetic block now precedes the real data
    orig = pd.to_datetime(synth.timestamp)
    assert (s_part.timestamp.dt.dayofweek.value_counts().sort_index().values ==
            orig.dt.dayofweek.value_counts().sort_index().values).all()            # shifted by whole weeks
    assert a["train"].timestamp.max() < a["val"].timestamp.min() <= a["val"].timestamp.max() < a["test"].timestamp.min()
    assert a["test"].timestamp.min() >= real.timestamp.max() - pd.Timedelta(days=retrain.HOLDOUT_DAYS)
    assert (a["test"].timestamp.max() - a["test"].timestamp.min()) < pd.Timedelta(days=retrain.HOLDOUT_DAYS)   # exactly the latest week


def test_enough_real_data_drops_synthetic_entirely(real_df, synth_csv):
    long_real = pd.concat([real_df, real_df.assign(timestamp=real_df.timestamp + pd.Timedelta(days=28))])
    a = assemble(long_real.drop(columns=["camera_status", "camera_age_seconds"]), pd.read_csv(synth_csv))
    assert a["mode"] == "live" and (a["df"].sample_weight == 1.0).all() and len(a["df"]) == len(long_real)


def test_champion_challenger_gate(tmp_path, real_df, synth_csv):
    st = _store_with(real_df)
    out_dir = tmp_path / "artifacts" / "pipelines"
    quiet = dict(synthetic_csv=synth_csv, params=TINY, log=lambda *_: None)

    dry = run_retrain(st, out_dir=out_dir, dry_run=True, **quiet)
    assert dry["decision"] == "would_promote" and not out_dir.exists()             # no champion yet; dry run writes nothing

    first = run_retrain(st, out_dir=out_dir, **quiet)
    assert first["decision"] == "promoted" and first["mode"] == "mixed"
    meta = json.loads((out_dir / "model_meta.json").read_text())
    assert meta["data_source"] == "mixed" and meta["version"] == first["version"]
    for f in ("xgb_multi_horizon.joblib", "residual_dist.joblib", "profile.joblib", "saturation_classifier.joblib"):
        assert (out_dir / f).exists()

    again = run_retrain(st, out_dir=out_dir, **quiet)                              # same data, same recipe: no real improvement
    assert again["decision"] == "kept_champion" and "not clearly better" in again["reason"]
    assert meta["version"] == json.loads((out_dir / "model_meta.json").read_text())["version"]
    history = json.loads((out_dir.parent / "registry.json").read_text())
    assert [h["decision"] for h in history] == ["promoted", "kept_champion"]
