from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.svm import LinearSVC


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    class_weight: Optional[dict[str, float]] = None,
    n_jobs: int = -1,
    tol: float = 1e-3,
    max_iter: int = 5000,
    **kwargs: Any,
) -> LinearSVC:
    """Train a linear SVM classifier with multi-core support via liblinear."""
    # n_jobs is passed directly so liblinear can parallelize one-vs-rest fits
    # (available in scikit-learn >=1.3). Fall back to setting the attribute for
    # compatibility with slightly older versions that still honor the field.
    model = LinearSVC(
        C=C,
        class_weight=class_weight,
        dual=True,
        loss="squared_hinge",
        tol=tol,
        max_iter=max_iter,
        fit_intercept=True,
        random_state=42,
        n_jobs=n_jobs,
        **kwargs,
    )
    if not hasattr(model, "n_jobs"):
        model.n_jobs = n_jobs

    model.fit(X_train, y_train)
    return model
