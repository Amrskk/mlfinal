"""A/B test of calibration choices on the stacking model.

Compares five configurations of the same fitted stack:
    A) uncalibrated, passthrough=False
    B) isotonic calibration on val
    C) Platt / sigmoid calibration on val
    D) passthrough=True, uncalibrated
    E) passthrough=True + sigmoid

Output: calibration_ablation.csv  (see README §Results for the table)
"""

import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, ".")
from src.feature_engineering import RAW_FEATURES, MagicFeatureEngineer  # noqa: E402
from src.metrics import partial_auc, tpr_at_fpr  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def make_stack(passthrough):
    base = [
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
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
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
        ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
    ]
    return StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        passthrough=passthrough,
        n_jobs=-1,
    )


def make_pipe(model):
    return ImbPipeline(
        [
            ("features", MagicFeatureEngineer(log_transform=True, keep_raw=True)),
            ("scale", StandardScaler()),
            ("smote", BorderlineSMOTE(random_state=SEED, k_neighbors=5)),
            ("model", model),
        ]
    )


def eval_probs(name, p, y_true):
    return {
        "config": name,
        "ROC-AUC": roc_auc_score(y_true, p),
        "PR-AUC": average_precision_score(y_true, p),
        "pAUC<=0.2": partial_auc(y_true, p, 0.2),
        "TPR@FPR=0.01": tpr_at_fpr(y_true, p, 0.01),
        "TPR@FPR=0.05": tpr_at_fpr(y_true, p, 0.05),
        "TPR@FPR=0.10": tpr_at_fpr(y_true, p, 0.10),
    }


def main():
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

    t0 = time.time()

    print("fitting stack (passthrough=False)...", flush=True)
    t = time.time()
    stack_no = make_pipe(make_stack(False)).fit(X_tr, y_tr)
    print(f"  done in {time.time() - t:.1f}s", flush=True)

    print("fitting stack (passthrough=True)...", flush=True)
    t = time.time()
    stack_yes = make_pipe(make_stack(True)).fit(X_tr, y_tr)
    print(f"  done in {time.time() - t:.1f}s", flush=True)

    rows = []
    rows.append(
        eval_probs(
            "A) stack uncalibrated (passthrough=False)",
            stack_no.predict_proba(X_te)[:, 1],
            y_te,
        )
    )

    cal_iso = CalibratedClassifierCV(FrozenEstimator(stack_no), method="isotonic")
    cal_iso.fit(X_va, y_va)
    rows.append(
        eval_probs(
            "B) stack + isotonic on val",
            cal_iso.predict_proba(X_te)[:, 1],
            y_te,
        )
    )

    cal_sig = CalibratedClassifierCV(FrozenEstimator(stack_no), method="sigmoid")
    cal_sig.fit(X_va, y_va)
    rows.append(
        eval_probs(
            "C) stack + sigmoid (Platt) on val",
            cal_sig.predict_proba(X_te)[:, 1],
            y_te,
        )
    )

    rows.append(
        eval_probs(
            "D) stack passthrough=True, uncalibrated",
            stack_yes.predict_proba(X_te)[:, 1],
            y_te,
        )
    )

    cal_sig_pt = CalibratedClassifierCV(FrozenEstimator(stack_yes), method="sigmoid")
    cal_sig_pt.fit(X_va, y_va)
    rows.append(
        eval_probs(
            "E) stack passthrough=True + sigmoid",
            cal_sig_pt.predict_proba(X_te)[:, 1],
            y_te,
        )
    )

    print("\n CALIBRATION ABLATION (test set)")
    df_out = pd.DataFrame(rows).set_index("config")
    print(df_out.round(4).to_string())
    df_out.round(4).to_csv("calibration_ablation.csv")
    print(f"\ntotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
