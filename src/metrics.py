"""Task-specific evaluation metrics for gamma/hadron discrimination.

The MAGIC analysis cares about the **low-FPR region** of the ROC curve,
not the aggregate AUC. These helpers compute:

  * tpr_at_fpr   : gamma signal efficiency at a fixed hadron-misID rate
  * partial_auc  : AUC restricted to FPR <= max_fpr (normalized to [0, 1])
  * efficiency_table : prints a summary at multiple operating points
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_fpr(y_true, y_score, fpr_target: float) -> float:
    """Return the TPR at the operating point with FPR <= fpr_target (largest such).

    Parameters

    y_true : array-like, binary {0, 1} where 1 = gamma (signal)
    y_score : array-like, predicted probability of class 1
    fpr_target : float, target false-positive (hadron mis-ID) rate
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # roc_curve returns increasing fpr; pick the largest fpr <= target
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    idx = max(0, idx)
    return float(tpr[idx])


def partial_auc(y_true, y_score, max_fpr: float = 0.2) -> float:
    """Partial AUC restricted to FPR <= max_fpr, normalized to [0, 1].

    Uses sklearn's built-in McClish-correction (`roc_auc_score` with `max_fpr`).
    """
    return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))


def efficiency_table(
    y_true,
    y_score,
    fpr_targets: Iterable[float] = (0.01, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Build a DataFrame of {fpr_target: tpr_at_that_point}.

    This is the headline metric for gamma/hadron classification.
    """
    rows = [
        {"fpr_target": f, "tpr (gamma efficiency)": tpr_at_fpr(y_true, y_score, f)}
        for f in fpr_targets
    ]
    return pd.DataFrame(rows)
