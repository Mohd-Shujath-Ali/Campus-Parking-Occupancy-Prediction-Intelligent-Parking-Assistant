"""Central loader for configs/model_config.yaml (single source of truth)."""
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


@lru_cache(maxsize=None)
def _load(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Return the parsed model_config.yaml (cached)."""
    return _load(str(path or CONFIG_DIR / "model_config.yaml"))


def load_zones_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Return the parsed zones.yaml (cached)."""
    return _load(str(path or CONFIG_DIR / "zones.yaml"))


TUNED_PATH = CONFIG_DIR / "tuned_params.yaml"


def xgb_params(section: str, tuned: bool = True) -> Dict[str, Any]:
    """
    Hyperparameters for a model block, e.g. 'xgboost_regressor'.

    Defaults come from model_config.yaml. If configs/tuned_params.yaml exists
    (written by experiments/tune_xgboost.py) its values override the defaults,
    so the tuned model is what training and serving use.
    """
    params = dict(load_model_config()["models"][section])
    if tuned and TUNED_PATH.exists():
        params.update(yaml.safe_load(TUNED_PATH.read_text()).get(section, {}))
    return params
