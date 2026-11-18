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


def xgb_params(section: str) -> Dict[str, Any]:
    """Copy of a hyperparameter block, e.g. 'xgboost_regressor'."""
    return dict(load_model_config()["models"][section])
