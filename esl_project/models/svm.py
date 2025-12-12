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
    max_iter: int = 2000,
    **kwargs: Any,
) -> LinearSVC:
    """Train a linear SVM classifier with multi-core support via liblinear."""
    model = LinearSVC(
        C=C,
        class_weight=class_weight,
        dual=True,
        loss="squared_hinge",
        tol=tol,
        max_iter=max_iter,
        fit_intercept=True,
        random_state=42,
        **kwargs,
    )
    # LinearSVC exposes n_jobs starting from scikit-learn 1.3;
    # set the attribute if available to unlock multi-core training.
    if hasattr(model, "n_jobs"):
        model.n_jobs = n_jobs

    model.fit(X_train, y_train)
    return model
