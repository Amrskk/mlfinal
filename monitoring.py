"""Drift and performance monitoring for the deployed classifier.

Three monitors, three remediations:

  1. Input drift           → PSI on each raw Hillas parameter.
                             PSI > 0.25 ⇒ retrain on a fresh labeled batch.
  2. Calibration drift     → Brier score + reliability curve gap.
                             Increase ⇒ re-fit sigmoid only.
  3. Operating-point drift → TPR @ FPR=0.01 measured on labeled holdout.
                             Drop ⇒ re-pick threshold against the holdout.

CLI usage
---------
  # input + (optional) labeled performance check
  python monitoring.py --reference telescope_data.csv \\
                       --current   recent_batch.csv \\
                       --labels    recent_labels.csv  \\
                       --model     artifacts/model_v1.joblib \\
                       --config    artifacts/deployment_config.json

  # input-only (no labels in production batch)
  python monitoring.py --reference telescope_data.csv \\
                       --current   recent_batch.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).parent))
from src.feature_engineering import RAW_FEATURES  # noqa: E402
from src.metrics import tpr_at_fpr  # noqa: E402

# !!! PSI


def psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Population Stability Index between two 1-D distributions.

    PSI = Σᵢ (cur_iᵢ - ref_iᵢ) · log(cur_iᵢ / ref_iᵢ)

    Rules of thumb:
        < 0.10  : no significant change
        0.10–0.25 : moderate shift (watch)
        > 0.25  : significant shift (retrain)
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    # Quantile-based bin edges from the reference distribution.
    # Using quantiles makes PSI robust to skewed features.
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        # Degenerate (near-constant) feature — call it 0.
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)

    # eps protects against log(0) when a bin is empty on one side
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] = RAW_FEATURES,
) -> pd.DataFrame:
    """Per-feature PSI table with a verdict column."""
    rows = []
    for f in features:
        s = psi(reference[f].values, current[f].values)
        if s > 0.25:
            verdict = "RETRAIN"
        elif s > 0.10:
            verdict = "watch"
        else:
            verdict = "ok"
        rows.append({"feature": f, "PSI": round(s, 4), "verdict": verdict})
    return pd.DataFrame(rows)


# !!! Performance & calibration monitors


def performance_report(
    y_true: np.ndarray,
    proba: np.ndarray,
    expected_tpr_at_fpr_001: float,
    tolerance: float = 0.05,
) -> dict:
    """Compare current TPR @ FPR=0.01 against the expected (training-time) value.

    A drop of more than `tolerance` triggers a re-pick-threshold action.
    """
    tpr_now = tpr_at_fpr(y_true, proba, 0.01)
    delta = tpr_now - expected_tpr_at_fpr_001
    if delta < -tolerance:
        verdict = "REPICK_THRESHOLD"
    elif delta < -tolerance / 2:
        verdict = "watch"
    else:
        verdict = "ok"
    return {
        "tpr_at_fpr_001_current": round(tpr_now, 4),
        "tpr_at_fpr_001_expected": round(expected_tpr_at_fpr_001, 4),
        "delta": round(delta, 4),
        "tolerance": tolerance,
        "verdict": verdict,
    }


def calibration_report(
    y_true: np.ndarray,
    proba: np.ndarray,
    expected_brier: float | None = None,
    tolerance: float = 0.02,
) -> dict:
    """Compare current Brier score against the training-time baseline.

    An increase larger than `tolerance` triggers a re-fit-sigmoid action.
    """
    brier_now = brier_score_loss(y_true, proba)
    if expected_brier is None:
        return {
            "brier_current": round(float(brier_now), 4),
            "brier_expected": None,
            "verdict": "no baseline",
        }
    delta = brier_now - expected_brier
    if delta > tolerance:
        verdict = "REFIT_SIGMOID"
    elif delta > tolerance / 2:
        verdict = "watch"
    else:
        verdict = "ok"
    return {
        "brier_current": round(float(brier_now), 4),
        "brier_expected": round(float(expected_brier), 4),
        "delta": round(float(delta), 4),
        "tolerance": tolerance,
        "verdict": verdict,
    }


# !!! CLI


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reference",
        required=True,
        help="CSV of the training distribution (telescope_data.csv).",
    )
    ap.add_argument(
        "--current",
        required=True,
        help="CSV of recent production inputs (same column layout as reference).",
    )
    ap.add_argument(
        "--labels",
        default=None,
        help=(
            "Optional CSV with a single 'class' column ('g'/'h') aligned to "
            "--current rows. Enables performance and calibration monitors."
        ),
    )
    ap.add_argument(
        "--model",
        default="artifacts/model_v1.joblib",
        help="Path to the deployed model artifact.",
    )
    ap.add_argument(
        "--config",
        default="artifacts/deployment_config.json",
        help="Path to deployment_config.json (gives the expected TPR baseline).",
    )
    ap.add_argument(
        "--expected-tpr",
        type=float,
        default=None,
        help=(
            "Optional override for the expected TPR @ FPR=0.01. "
            "Defaults to value in deployment_config.json if present."
        ),
    )
    args = ap.parse_args(argv)

    ref = pd.read_csv(args.reference, index_col=0)
    # The reference file ships with a `class` column; drop it for input PSI.
    if "class" in ref.columns:
        ref = ref.drop(columns=["class"])
    if not Path(args.current).exists():
        ap.error(f"--current file {args.current} not found")
    cur = pd.read_csv(args.current, index_col=0)
    if "class" in cur.columns:
        cur = cur.drop(columns=["class"])

    print("\n=== INPUT DRIFT (PSI per Hillas parameter) ===")
    psi_df = psi_report(ref, cur)
    print(psi_df.to_string(index=False))
    # Severity order, not lexicographic: RETRAIN > watch > ok
    _severity = {"ok": 0, "watch": 1, "RETRAIN": 2}
    worst = max(psi_df["verdict"], key=_severity.get)
    summary = {"input_drift_overall": worst}
    if (psi_df["verdict"] == "RETRAIN").any():
        print("\n→ At least one feature exceeds PSI 0.25. Recommend RETRAIN.")
    elif (psi_df["verdict"] == "watch").any():
        print("\n→ Some features in the watch range (0.10–0.25). Keep monitoring.")
    else:
        print("\n→ Inputs look stable.")

    # If labels are provided, run performance + calibration monitors.
    if args.labels:
        labels_df = pd.read_csv(args.labels)
        if "class" not in labels_df.columns:
            ap.error("--labels file must have a 'class' column ('g' or 'h').")
        y_true = (labels_df["class"] == "g").astype(int).values
        if len(y_true) != len(cur):
            ap.error(
                f"label count ({len(y_true)}) does not match current rows ({len(cur)})"
            )

        model = joblib.load(args.model)
        proba = model.predict_proba(cur[RAW_FEATURES])[:, 1]

        # Baselines from config (set by run_full.py at training time)
        with open(args.config) as f:
            cfg = json.load(f)
        expected_tpr = (
            args.expected_tpr
            if args.expected_tpr is not None
            else cfg.get("expected_tpr_at_fpr_001")
        )

        print("\n=== PERFORMANCE DRIFT (TPR @ FPR=0.01) ===")
        if expected_tpr is None:
            print(
                "No expected_tpr_at_fpr_001 baseline in config. "
                "Pass --expected-tpr to enable verdict."
            )
            print(f"current TPR @ FPR=0.01 = {tpr_at_fpr(y_true, proba, 0.01):.4f}")
        else:
            perf = performance_report(y_true, proba, expected_tpr)
            print(json.dumps(perf, indent=2))
            summary["performance"] = perf["verdict"]

        print("\n CALIBRATION DRIFT (Brier score)")
        cal = calibration_report(
            y_true, proba, expected_brier=cfg.get("expected_brier")
        )
        print(json.dumps(cal, indent=2))
        summary["calibration"] = cal["verdict"]

    print("\n SUMMARY ")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
