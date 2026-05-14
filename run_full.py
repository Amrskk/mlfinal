"""Full-config run that produces the headline numbers for the README.

Mirrors `magic_gamma_pipeline.ipynb`:
  - 60/20/20 stratified train/val/test split
  - Optuna-tuned XGBoost (25 TPE trials maximizing pAUC <= 0.2 on val)
  - Sigmoid calibration on the validation split
  - Stacking comparison with 5 base learners (LGBM, tuned XGB, RF, SVM-RBF, LR)

Writes:
  - results.csv         : the headline table
  - results.json        : same data as JSON
  - best_xgb_params.json: tuned hyperparameters for downstream reuse
"""

import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, ".")
from src.feature_engineering import RAW_FEATURES, MagicFeatureEngineer  # noqa: E402
from src.metrics import partial_auc, tpr_at_fpr  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

USE_MLFLOW = os.environ.get("MLFLOW_DISABLED", "0") != "1"
try:
    if USE_MLFLOW:
        import mlflow  # type: ignore
        import mlflow.sklearn  # type: ignore
except ImportError:
    USE_MLFLOW = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def make_pipe(model):
    return ImbPipeline(
        [
            ("features", MagicFeatureEngineer(log_transform=True, keep_raw=True)),
            ("scale", StandardScaler()),
            ("smote", BorderlineSMOTE(random_state=SEED, k_neighbors=5)),
            ("model", model),
        ]
    )


def score_block(y_true, p):
    return {
        "ROC-AUC": roc_auc_score(y_true, p),
        "PR-AUC": average_precision_score(y_true, p),
        "pAUC<=0.2": partial_auc(y_true, p, 0.2),
        "TPR@FPR=0.01": tpr_at_fpr(y_true, p, 0.01),
        "TPR@FPR=0.05": tpr_at_fpr(y_true, p, 0.05),
        "TPR@FPR=0.10": tpr_at_fpr(y_true, p, 0.10),
    }


def tune_xgb(X_tr, y_tr, X_va, y_va, n_trials=25):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        m = XGBClassifier(
            **params,
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED,
            verbosity=0,
        )
        pipe = make_pipe(m).fit(X_tr, y_tr)
        p = pipe.predict_proba(X_va)[:, 1]
        return partial_auc(y_va, p, 0.2)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def _maybe_start_mlflow():
    """Configure MLflow with a local file store and start a run.

    Returns the active run (or None if MLflow is disabled / unavailable).
    The file store keeps everything under ./mlruns so no separate server
    is needed; `mlflow ui` works straight against it.
    """
    if not USE_MLFLOW:
        return None
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("magic-gamma")
    return mlflow.start_run(run_name="xgboost-optuna-headline")


def main():
    t0 = time.time()
    mlflow_run = _maybe_start_mlflow()

    df = (
        pd.read_csv("telescope_data.csv", index_col=0)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    y = (df["class"] == "g").astype(int).values
    X = df[RAW_FEATURES].copy()
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=SEED
    )
    print(
        f"split  train {X_tr.shape[0]}  val {X_va.shape[0]}  test {X_te.shape[0]}",
        flush=True,
    )

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)

    print("tuning XGBoost via Optuna (25 trials)...", flush=True)
    t = time.time()
    best_xgb_params, best_val = tune_xgb(X_tr, y_tr, X_va, y_va, n_trials=25)
    print(
        f"  Optuna done in {time.time() - t:.1f}s, "
        f"best pAUC<=0.2 (val) = {best_val:.4f}",
        flush=True,
    )
    print(f"  best params: {best_xgb_params}", flush=True)

    baselines = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
        ),
        "XGBoost (tuned)": XGBClassifier(
            **best_xgb_params,
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=800,
            num_leaves=63,
            learning_rate=0.05,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            random_state=SEED,
            verbose=-1,
        ),
        "SVM-RBF": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=SEED,
        ),
    }

    summary = {}
    for name, model in baselines.items():
        t = time.time()
        pipe = make_pipe(model).fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        summary[name] = score_block(y_te, p)
        summary[name]["fit_s"] = round(time.time() - t, 1)
        print(f"  {name:18s} done in {summary[name]['fit_s']}s", flush=True)

    # HEADLINE PATH: fit + sigmoid-calibrate the tuned XGBoost, pick a
    # deployment threshold at FPR=0.01 on val, save artifacts
    print("fitting headline XGBoost...", flush=True)
    t = time.time()
    headline_xgb_pipe = make_pipe(
        XGBClassifier(
            **best_xgb_params,
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED,
            verbosity=0,
        )
    ).fit(X_tr, y_tr)
    print(f"  headline fit in {time.time() - t:.1f}s", flush=True)

    print("sigmoid-calibrating headline on val...", flush=True)
    headline_cal = CalibratedClassifierCV(
        FrozenEstimator(headline_xgb_pipe), method="sigmoid"
    )
    headline_cal.fit(X_va, y_va)

    # Pick deployment threshold on VAL at FPR=0.01 (operational target).
    # Using val (not test) for threshold pick keeps test sacred.
    p_va = headline_cal.predict_proba(X_va)[:, 1]
    fpr_va, tpr_va, thr_va = roc_curve(y_va, p_va)
    idx = int(np.searchsorted(fpr_va, 0.01, side="right") - 1)
    idx = max(0, idx)
    deploy_threshold = float(thr_va[idx])
    print(
        f"  deployment threshold = {deploy_threshold:.4f} "
        f"(val FPR={fpr_va[idx]:.4f}, val TPR={tpr_va[idx]:.4f})",
        flush=True,
    )

    # Save the deployable artifact
    joblib.dump(headline_cal, artifact_dir / "model_v1.joblib")
    # Compute baselines for the monitoring module to compare against.
    # These are measured on the validation split — the same data the
    # threshold was picked against — so production drift can be detected
    # relative to "the model's known good behavior at deploy time."
    expected_tpr_at_fpr_001 = float(tpr_va[idx])
    expected_brier = float(brier_score_loss(y_va, p_va))
    deployment_config = {
        "model_path": "artifacts/model_v1.joblib",
        "model_version": "v1",
        "model_type": "xgboost-optuna-tuned + sigmoid",
        "feature_order": list(RAW_FEATURES),
        "deployment_threshold": deploy_threshold,
        "threshold_chosen_at": "val FPR=0.01",
        "training_seed": SEED,
        "training_rows": int(X_tr.shape[0]),
        "validation_rows": int(X_va.shape[0]),
        "best_xgb_params": best_xgb_params,
        # Baselines for the monitoring module
        "expected_tpr_at_fpr_001": expected_tpr_at_fpr_001,
        "expected_brier": expected_brier,
    }
    with open(artifact_dir / "deployment_config.json", "w") as f:
        json.dump(deployment_config, f, indent=2)
    print(
        "  saved artifacts/model_v1.joblib + artifacts/deployment_config.json",
        flush=True,
    )

    p_te_headline = headline_cal.predict_proba(X_te)[:, 1]
    summary["XGBoost+sigmoid (DEPLOYED)"] = score_block(y_te, p_te_headline)

    print("fitting stack (5 base learners)...", flush=True)
    t = time.time()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    base_learners = [
        (
            "lgbm",
            LGBMClassifier(
                n_estimators=800,
                num_leaves=63,
                learning_rate=0.05,
                min_child_samples=20,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=-1,
                random_state=SEED,
                verbose=-1,
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                **best_xgb_params,
                eval_metric="auc",
                tree_method="hist",
                n_jobs=-1,
                random_state=SEED,
                verbosity=0,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=SEED,
            ),
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=SEED,
            ),
        ),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
    ]
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        cv=cv,
        passthrough=False,
        n_jobs=-1,
    )
    stacking_pipe = make_pipe(stack).fit(X_tr, y_tr)
    print(f"  stack fit in {time.time() - t:.1f}s", flush=True)

    print("calibrating (sigmoid)...", flush=True)
    t = time.time()
    cal = CalibratedClassifierCV(FrozenEstimator(stacking_pipe), method="sigmoid")
    cal.fit(X_va, y_va)
    print(f"  calibration in {time.time() - t:.1f}s", flush=True)

    p_te = cal.predict_proba(X_te)[:, 1]
    prec, rec, thr = precision_recall_curve(y_te, p_te)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.argmax(f1s[:-1]))
    summary["Stack + sigmoid (5 base)"] = score_block(y_te, p_te)
    summary["Stack + sigmoid (5 base)"]["BestF1"] = float(f1s[best_idx])
    summary["Stack + sigmoid (5 base)"]["BestF1Thresh"] = float(thr[best_idx])

    results = pd.DataFrame(summary).T
    print("\n FINAL TEST-SET RESULTS ")
    print(results.round(4).to_string())
    results.round(4).to_csv("results.csv")
    with open("results.json", "w") as f:
        json.dump(
            {k: {kk: float(vv) for kk, vv in v.items()} for k, v in summary.items()},
            f,
            indent=2,
        )
    with open("best_xgb_params.json", "w") as f:
        json.dump(best_xgb_params, f, indent=2)

    # MLflow logging
    if mlflow_run is not None:
        mlflow.log_params(best_xgb_params)
        mlflow.log_param("model_type", deployment_config["model_type"])
        mlflow.log_param("training_rows", deployment_config["training_rows"])
        mlflow.log_param("deployment_threshold", deploy_threshold)

        # Headline metrics (the DEPLOYED model row). MLflow restricts metric
        # names to alnum + _ - . space : / — sanitize @ = < > out.
        def _mlflow_safe(name):
            return (
                name.replace("<=", "_lte_")  # two-char first
                .replace(">=", "_gte_")
                .replace("<", "_lt_")  # now < is guaranteed standalone
                .replace(">", "_gt_")
                .replace("@", "_at_")
                .replace("=", "_eq_")
            )

        for k, v in summary["XGBoost+sigmoid (DEPLOYED)"].items():
            mlflow.log_metric(_mlflow_safe(k), v)
        mlflow.log_artifact(str(artifact_dir / "model_v1.joblib"))
        mlflow.log_artifact(str(artifact_dir / "deployment_config.json"))
        mlflow.log_artifact("results.csv")
        mlflow.sklearn.log_model(
            headline_cal,
            artifact_path="model",
            registered_model_name="magic-gamma",
        )
        mlflow.end_run()
        print("  mlflow run logged to ./mlruns")

    print(f"\ntotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
