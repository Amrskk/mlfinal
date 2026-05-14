"""Smoke test — verify the headline XGBoost pipeline plus a quick stack check.

Run from the repo root:
    python smoke_test.py

Uses small estimator counts so it completes in ~30s. The real numbers
come from `run_full.py`; this is just plumbing verification.
"""

import os
import random
import sys
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
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, ".")
from src.feature_engineering import (  # noqa: E402
    RAW_FEATURES,
    MagicFeatureEngineer,
    validate_physical_constraints,
)
from src.metrics import efficiency_table, partial_auc  # noqa: E402

warnings.filterwarnings("ignore")

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


# 1) Load & dedup
df = pd.read_csv("telescope_data.csv", index_col=0)
print(f"raw {df.shape},  dupes {df.duplicated().sum()}")
df = df.drop_duplicates().reset_index(drop=True)
y = (df["class"] == "g").astype(int).values
X = df[RAW_FEATURES].copy()

# 2) Stratified 60/20/20 split
X_tv, X_te, y_tv, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=SEED
)
print(f"train {X_tr.shape}  val {X_va.shape}  test {X_te.shape}")

# 3) Feature engineering sanity check
fe = MagicFeatureEngineer()
Xt = fe.fit_transform(X_tr)
print(f"inf? {np.isinf(Xt).any()}  nan? {np.isnan(Xt).any()}  shape {Xt.shape}")
print(
    f"physical constraint pass rate: {validate_physical_constraints(X_tr).mean():.4f}"
)

# 4) Headline path — fit tuned-ish XGBoost (defaults, no Optuna in smoke test)
print("\n HEADLINE PATH: XGBoost + sigmoid calibration ")
xgb_pipe = make_pipe(
    XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
        verbosity=0,
    )
).fit(X_tr, y_tr)
xgb_cal = CalibratedClassifierCV(FrozenEstimator(xgb_pipe), method="sigmoid")
xgb_cal.fit(X_va, y_va)
p_xgb = xgb_cal.predict_proba(X_te)[:, 1]
print(f"ROC-AUC          : {roc_auc_score(y_te, p_xgb):.4f}")
print(f"Partial AUC (.2) : {partial_auc(y_te, p_xgb, 0.2):.4f}")
print(f"PR-AUC           : {average_precision_score(y_te, p_xgb):.4f}")
print(efficiency_table(y_te, p_xgb).round(4).to_string(index=False))

# 5) Quick stack plumbing check — smaller estimators for speed
print("\n STACK PLUMBING CHECK (smaller estimators) ")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
base = [
    (
        "lgbm",
        LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.07,
            n_jobs=-1,
            random_state=SEED,
            verbose=-1,
        ),
    ),
    (
        "xgb",
        XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.07,
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
            n_estimators=200,
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
    estimators=base,
    final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
    cv=cv,
    passthrough=False,
    n_jobs=-1,
)
stack_pipe = make_pipe(stack).fit(X_tr, y_tr)
stack_cal = CalibratedClassifierCV(FrozenEstimator(stack_pipe), method="sigmoid")
stack_cal.fit(X_va, y_va)
p_stack = stack_cal.predict_proba(X_te)[:, 1]
print(f"ROC-AUC          : {roc_auc_score(y_te, p_stack):.4f}")
print(f"Partial AUC (.2) : {partial_auc(y_te, p_stack, 0.2):.4f}")
print(efficiency_table(y_te, p_stack).round(4).to_string(index=False))
