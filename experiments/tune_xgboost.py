"""
Hyperparameter optimisation (rubric 3.7) for the +15 min XGBoost forecaster.

* Objective: mean MAE over TimeSeriesSplit folds on the TRAIN period only
  (validation and test are never seen by the search).
* Bayesian optimisation (Optuna TPE) vs a random-search baseline with the same
  trial budget, so the benefit of guided search is measured, not assumed.
* Folds are chronological with a gap of cv_gap steps x zones rows, so labels of
  late training rows never overlap the validation fold.
* Writes configs/tuned_params.yaml, which training and serving pick up.

Run:  python -m experiments.tune_xgboost [--trials 30]
"""
import argparse
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from evaluation.metrics import compute_regression_metrics
from models.data_prep import cv_gap_rows, load_splits
from models.pipeline import build_pipeline
from utils.config import TUNED_PATH, load_model_config, xgb_params

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def make_model(params: dict):
    base = xgb_params("xgboost_regressor", tuned=False)
    base.update(params)
    return build_pipeline(xgb.XGBRegressor(**base))


def main(n_trials: int):
    sp = load_splits()
    tr, va, te = sp["train"], sp["val"], sp["test"]
    cv = TimeSeriesSplit(n_splits=load_model_config()["training"]["cv_n_splits"],
                         gap=cv_gap_rows())

    def objective(trial):
        model = make_model(suggest(trial))
        # neg MAE -> MAE
        return -cross_val_score(model, tr.X, tr.y["H15"], cv=cv,
                                scoring="neg_mean_absolute_error", n_jobs=1).mean()

    studies = {}
    for name, sampler in (("Bayesian (TPE)", optuna.samplers.TPESampler(seed=42)),
                          ("Random search", optuna.samplers.RandomSampler(seed=42))):
        st = optuna.create_study(direction="minimize", sampler=sampler)
        st.optimize(objective, n_trials=n_trials)
        studies[name] = st
        print(f"{name:15s} best CV MAE = {st.best_value:.4f}")

    best_name = min(studies, key=lambda k: studies[k].best_value)
    best = studies[best_name].best_params
    print(f"\nwinner: {best_name}\n{json.dumps(best, indent=2)}")

    # Default vs tuned on the untouched validation and test periods
    rows = []
    for label, params in (("Default (model_config.yaml)", {}), (f"Tuned ({best_name})", best)):
        m = make_model(params).fit(tr.X, tr.y["H15"])
        for sn, s in (("val", va), ("test", te)):
            rows.append({"Model": label, "Split": sn,
                         **compute_regression_metrics(s.y["H15"].values, m.predict(s.X))})
    cmp = pd.DataFrame(rows)
    print("\n", cmp.to_string(index=False))

    # Persist artefacts
    pd.concat([s.trials_dataframe().assign(search=n) for n, s in studies.items()]
              ).to_csv("artifacts/tuning/xgb_h15_trials.csv", index=False)
    cmp.to_csv("artifacts/tuning/default_vs_tuned.csv", index=False)
    TUNED_PATH.write_text(
        "# Written by experiments/tune_xgboost.py - overrides model_config.yaml\n"
        + yaml.safe_dump({"xgboost_regressor": best}, sort_keys=True))

    # Search history plot: best-so-far CV MAE per trial
    fig, ax = plt.subplots(figsize=(7, 4))
    for n, s in studies.items():
        vals = [t.value for t in s.trials]
        ax.plot(np.minimum.accumulate(vals), label=n, lw=2)
    ax.set_xlabel("trial"); ax.set_ylabel("best CV MAE so far (cars)")
    ax.set_title("Hyperparameter search: Bayesian vs random"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("artifacts/figures/tuning_history.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=30)
    main(ap.parse_args().trials)
