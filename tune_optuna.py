#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optuna tuner for scratch & scikit models.

Usage
-----
python optuna_tune.py \
  -f Data-20251001/ozone_complet.txt \
  -t r \
  -F maxO3 \
  -a RandomForest \
  --lib scratch \
  --n-trials 50

python optuna_tune.py \
  -f Data-20251001/Carseats_prepared.csv \
  -t c \
  -F High \
  -a SVM \
  --lib scikit \
  --n-trials 50

Notes
-----
- Uses your existing utils.read_file_wtype, utils.get_class / utils.algos_sci_map,
  and bench.benchmark_* for consistent preprocessing & metrics.
- For classification: maximizes best F1 (with threshold search from bench).
- For regression:   maximizes R².
- Data scaling (train mean/std) matches your main.py.
"""

import argparse
import numpy as np
import optuna
import utils
import bench

# ----------------------------- arg parsing -----------------------------

parser = argparse.ArgumentParser(
    description="Hyperparameter optimization with Optuna (scratch & scikit).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-f", "--file", required=True, help="Input file path")
parser.add_argument("-t", "--type", choices=["r", "regression", "c", "classification"], default="r",
                    help="Task type")
parser.add_argument("-F", "--to-find", required=True, help="Target/label column")
parser.add_argument("-a", "--algorithm", choices=["DecisionTree", "RandomForest", "Ridge", "Lasso", "SVM"],
                    required=True, help="Algorithm to tune")
parser.add_argument("--lib", choices=["scratch", "scikit"], default="scratch",
                    help="Tune scratch or scikit implementation")
parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
parser.add_argument("--timeout", type=int, default=None, help="Global timeout in seconds")
parser.add_argument("--seed", type=int, default=0, help="Random seed for split & Optuna")
args = parser.parse_args()

typ = args.type[0]  # normalize to 'r' or 'c'

# ----------------------------- search spaces -----------------------------

def suggest_params(trial: optuna.Trial, algo: str, typ: str, lib: str):
    """
    Returns well-rounded, clean hyperparameters:
    - Integers sampled with fixed step size
    - Floats sampled from discrete grids (log 1-2-5 or 1-3-10 patterns)
    to avoid messy decimal values.
    """

    LOG_1_2_5 = lambda lo, hi: [v for v in
        [m * 10**e for e in range(-6, 7) for m in (1, 2, 5)] if lo <= v <= hi]
    LOG_1_3_10 = lambda lo, hi: [v for v in
        [m * 10**e for e in range(-6, 7) for m in (1, 3, 10)] if lo <= v <= hi]

    GR_C      = LOG_1_2_5(1e-1, 1e2)          # C ∈ {0.1,0.2,0.5,1,2,5,10,20,50,100}
    GR_LR     = [0.00001, 0.0001, 0.001, 0.01, 0.1]  # learning_rate
    GR_EPS    = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0] # epsilon
    GR_ALPHA  = LOG_1_3_10(1e-3, 1e3)         # alpha Ridge/Lasso
    GR_L1     = LOG_1_3_10(1e-4, 1e-1)        # l1_penalty
    GR_TOL    = LOG_1_3_10(1e-5, 1e-2)        # tol
    GR_MIN_GAIN = LOG_1_3_10(1e-7, 1e-4)      # min_gain

    # ---------------- Decision Tree ----------------
    if algo == "DecisionTree":
        if typ == "r":
            if lib == "scratch":
                return {
                    "max_depth": trial.suggest_int("max_depth", 3, 12, step=1),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20, step=1),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10, step=1),
                    "max_splits_per_feature": trial.suggest_categorical(
                        "max_splits_per_feature", [64, 128, 256, 512, None]
                    ),
                    "min_gain": trial.suggest_categorical("min_gain", GR_MIN_GAIN),
                }
            else:  # scikit
                return {
                    "criterion": trial.suggest_categorical("criterion", ["squared_error"]),
                    "max_depth": trial.suggest_int("max_depth", 3, 12, step=1),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20, step=1),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10, step=1),
                    "random_state": args.seed,
                }
        else:  # typ == "c"
            if lib == "scratch":
                return {
                    "max_depth": trial.suggest_int("max_depth", 3, 12, step=1),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20, step=1),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10, step=1),
                    "max_thresholds": trial.suggest_categorical("max_thresholds", [32, 64, 128, 256, None]),
                    "min_gain": trial.suggest_categorical("min_gain", GR_MIN_GAIN),
                }
            else:  # scikit
                return {
                    "criterion": trial.suggest_categorical("criterion", ["entropy", "gini"]),
                    "max_depth": trial.suggest_int("max_depth", 3, 12, step=1),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20, step=1),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10, step=1),
                    "random_state": args.seed,
                }

    # ---------------- Random Forest ----------------
    if algo == "RandomForest":
        if typ == "r":
            if lib == "scratch":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                    "max_depth": trial.suggest_categorical("max_depth", [None, 6, 8, 10, 12]),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 12, step=1),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                    "random_state": args.seed,
                    "n_jobs": -1,
                }
            else:  # scikit
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                    "criterion": "squared_error",
                    "max_depth": trial.suggest_categorical("max_depth", [None, 6, 8, 10, 12]),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 12, step=1),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6, step=1),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                    "random_state": args.seed,
                    "n_jobs": -1,
                }
        else:  # typ == "c"
            if lib == "scratch":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                    "max_depth": trial.suggest_int("max_depth", 6, 14, step=1),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 12, step=1),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6, step=1),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                    "random_state": args.seed,
                    "n_jobs": -1,
                }
            else:  # scikit
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                    "criterion": trial.suggest_categorical("criterion", ["entropy", "gini"]),
                    "max_depth": trial.suggest_int("max_depth", 6, 14, step=1),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 12, step=1),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6, step=1),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                    "random_state": args.seed,
                    "n_jobs": -1,
                }

    # ---------------- Ridge ----------------
    if algo == "Ridge":
        if lib == "scratch":
            return {"alpha": trial.suggest_categorical("alpha", GR_ALPHA)}
        else:
            return {"alpha": trial.suggest_categorical("alpha", GR_ALPHA), "random_state": args.seed}

    # ---------------- Lasso ----------------
    if algo == "Lasso":
        if lib == "scratch":
            return {
                "learning_rate": trial.suggest_categorical("learning_rate", GR_LR),
                "l1_penalty": trial.suggest_categorical("l1_penalty", GR_L1),
                "max_iter": trial.suggest_int("max_iter", 200, 2000, step=50),
                "tol": trial.suggest_categorical("tol", GR_TOL),
                "fit_intercept": True,
                "normalize": True,
                "dtype": "float32",
            }
        else:  # scikit
            return {
                "alpha": trial.suggest_categorical("alpha", LOG_1_3_10(1e-4, 1e-1)),
                "max_iter": trial.suggest_int("max_iter", 500, 5000, step=100),
                "tol": trial.suggest_categorical("tol", LOG_1_3_10(1e-6, 1e-3)),
                "fit_intercept": True,
                "random_state": args.seed,
            }

    # ---------------- SVM ----------------
    if algo == "SVM":
        if typ == "c":
            if lib == "scratch":  # SVC
                return {
                    "learning_rate": trial.suggest_categorical("learning_rate", GR_LR),
                    "C": trial.suggest_categorical("C", GR_C),
                    "n_iters": trial.suggest_int("n_iters", 100, 800, step=50),
                    "shuffle": True,
                    "random_state": args.seed,
                    "dtype": "float32",
                }
            else:  # scikit LinearSVC
                return {
                    "C": trial.suggest_categorical("C", GR_C),
                    "loss": trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
                    "fit_intercept": True,
                    "max_iter": trial.suggest_int("max_iter", 1000, 20000, step=500),
                    "random_state": args.seed,
                }
        else:  # typ == "r" (SVR)
            if lib == "scratch":
                return {
                    "learning_rate": trial.suggest_categorical("learning_rate", GR_LR),
                    "C": trial.suggest_categorical("C", GR_C),
                    "epsilon": trial.suggest_categorical("epsilon", GR_EPS),
                    "n_iters": trial.suggest_int("n_iters", 100, 800, step=50),
                    "shuffle": True,
                    "random_state": args.seed,
                }
            else:  # scikit SVR (kernel linéaire pour comparabilité)
                return {
                    "kernel": "linear",
                    "C": trial.suggest_categorical("C", GR_C),
                    "epsilon": trial.suggest_categorical("epsilon", GR_EPS),
                    "max_iter": trial.suggest_int("max_iter", 1000, 20000, step=500),
                }

    raise ValueError(f"No search space for algo={algo}, typ={typ}, lib={lib}")

# ----------------------------- objective -----------------------------

def build_model_and_apply_params(algo: str, typ: str, lib: str, par: dict):
    """
    Create model (scratch or scikit) and push params dict into it.
    """
    if lib == "scratch":
        ModelClass = utils.get_class(algo, typ)
        model = ModelClass()
    else:
        model = utils.algos_sci_map[algo][typ]()  # scikit class

    # Best-effort set_params or setattr
    if hasattr(model, "set_params"):
        try:
            model.set_params(**par)
        except Exception:
            # fallback to setattr for any leftover keys
            for k, v in par.items():
                try:
                    setattr(model, k, v)
                except Exception:
                    pass
    else:
        for k, v in par.items():
            try:
                setattr(model, k, v)
            except Exception:
                pass
    return model

def objective(trial: optuna.Trial) -> float:
    # Read & prep data once per trial (fast enough; you can cache outside if desired)
    df = utils.read_file_wtype(args.file, typ)
    y = df[args.to_find]
    X = (df.drop(columns=[args.to_find])
           .select_dtypes(include=[np.number])
           .to_numpy(dtype=float, copy=True))

    # split
    X_train, X_test, y_train, y_test = utils.split(X, y, test_size=0.2, random_state=args.seed)

    # standardize like main.py
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_train = (X_train - mu) / sigma
    X_test  = (X_test  - mu) / sigma

    # params for this trial
    par = suggest_params(trial, args.algorithm, typ, args.lib)

    # build model with params
    model = build_model_and_apply_params(args.algorithm, typ, args.lib, par)

    # run bench and compute score
    if typ == "c":
        res = bench.benchmark_classification(model, X_train, y_train, X_test, y_test)
        # maximize best F1 (after threshold search)
        score = float(res["best_metrics"]["f1"])
    else:
        res = bench.benchmark_regression(model, X_train, y_train, X_test, y_test)
        # maximize R^2
        score = float(res["reg_metrics"]["r2"])

    # you can log timing or metrics in trial for analysis
    trial.set_user_attr("times", res.get("times", {}))
    return score

# ----------------------------- run study -----------------------------

if __name__ == "__main__":
    study = optuna.create_study(
        study_name=f"{args.algorithm}-{typ}-{args.lib}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    print("\nBest trial:")
    bt = study.best_trial
    print(f"  value: {bt.value:.6f}")
    print("  params:")
    for k, v in bt.params.items():
        print(f"    {k}: {v}")

    # Optional: show a compact dict to copy/paste into params.yaml if you like
    print("\nYAML-like snippet to paste under params:")
    print(f"{args.algorithm}:")
    print(f"  {args.lib}:")
    print(f"    {typ}:")
    for k, v in bt.params.items():
        # basic formatting (floats vs ints)
        if isinstance(v, float):
            print(f"      {k}: {v:.6g}")
        else:
            print(f"      {k}: {v}")
