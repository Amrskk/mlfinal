"""Physics-informed feature engineering for the MAGIC Gamma Telescope.

All ratio features use an additive epsilon to avoid divide-by-zero on the
98 rows where fWidth == 0 and the 5 rows where fAlpha == 0. The original
Hillas inputs are also optionally log-transformed before being passed on,
because fLength, fWidth, fAsym, fM3*, and fDist are heavy-tailed.

This is implemented as a sklearn-compatible transformer so it composes
cleanly with Pipeline / StackingClassifier / cross_val_score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

EPS = 1e-3

RAW_FEATURES = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
]

ENGINEERED_FEATURES = [
    "ellipticity",  # fLength / (fWidth + eps)
    "shower_density",  # fSize / (fLength * fWidth + eps)
    "miss_parameter",  # fDist * sin(fAlpha)
    "concentration_ratio",  # fConc / (fConc1 + eps)
    "m3_magnitude",  # sqrt(fM3Long^2 + fM3Trans^2)
    "log_bright_pair",  # fSize + log10(fConc + eps)  -- I5 fix
    "long_asym",  # fAsym / (fLength + eps)
    "cos_alpha",  # cos(fAlpha)  -- N2 add
]


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    """log1p that preserves sign for symmetric, signed inputs (fAsym, fM3*)."""
    return np.sign(x) * np.log1p(np.abs(x))


class MagicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Compute physics-informed features. Pass-through preserves raw features.

    Parameters

    log_transform : bool, default=True
        If True, apply log1p (signed for fAsym/fM3*) to heavy-tailed inputs.
    keep_raw : bool, default=True
        If True, include the (optionally log-transformed) raw features in
        the output. If False, return only engineered features.
    """

    def __init__(self, log_transform: bool = True, keep_raw: bool = True):
        self.log_transform = log_transform
        self.keep_raw = keep_raw

    def fit(self, X, y=None):
        # stateless
        self.feature_names_in_ = (
            list(X.columns) if hasattr(X, "columns") else RAW_FEATURES
        )
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=RAW_FEATURES)
        else:
            X = X.copy()

        # Engineered features always computed from the original (unlogged) values
        # so the physics interpretation is preserved.
        eng = pd.DataFrame(index=X.index)
        eng["ellipticity"] = X["fLength"] / (X["fWidth"] + EPS)
        eng["shower_density"] = X["fSize"] / (X["fLength"] * X["fWidth"] + EPS)
        eng["miss_parameter"] = X["fDist"] * np.sin(np.deg2rad(X["fAlpha"]))
        eng["concentration_ratio"] = X["fConc"] / (X["fConc1"] + EPS)
        eng["m3_magnitude"] = np.sqrt(X["fM3Long"] ** 2 + X["fM3Trans"] ** 2)
        eng["log_bright_pair"] = X["fSize"] + np.log10(X["fConc"] + EPS)
        eng["long_asym"] = X["fAsym"] / (X["fLength"] + EPS)
        eng["cos_alpha"] = np.cos(np.deg2rad(X["fAlpha"]))

        if not self.keep_raw:
            return eng.values

        raw = X[RAW_FEATURES].copy()
        if self.log_transform:
            # positive heavy-tailed
            for col in ("fLength", "fWidth", "fDist"):
                raw[col] = np.log1p(raw[col])
            # signed heavy-tailed
            for col in ("fAsym", "fM3Long", "fM3Trans"):
                raw[col] = _signed_log1p(raw[col])
            # fSize is already log10(...), don't transform again
            # fConc, fConc1 in [0,1], leave
            # fAlpha in [0,90], leave (cos_alpha already captures it)
        out = pd.concat([raw, eng], axis=1)
        return out.values

    def get_feature_names_out(self, input_features=None):
        if self.keep_raw:
            return np.array(RAW_FEATURES + ENGINEERED_FEATURES)
        return np.array(ENGINEERED_FEATURES)


def validate_physical_constraints(X: pd.DataFrame) -> pd.Series:
    """Return a boolean mask of rows that satisfy known physical constraints.

    Constraints come from the Hillas-parameter definitions. Note: B3 fix —
    we relax `fWidth > 0` to `fWidth >= 0` because 98 real samples have
    fWidth exactly 0 and would otherwise be rejected.
    """
    mask = (
        (X["fLength"] > 0)
        & (X["fWidth"] >= 0)
        & (X["fLength"] >= X["fWidth"])
        & (X["fConc"].between(0, 1))
        & (X["fConc1"].between(0, 1))
        & (X["fConc"] >= X["fConc1"])
        & (X["fAlpha"].between(0, 90))
        & (X["fDist"] > 0)
    )
    return mask
