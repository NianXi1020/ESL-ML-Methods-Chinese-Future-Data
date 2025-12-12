from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits


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
    # Explicitly control thread usage via threadpoolctl so liblinear can
    # parallelize across CPU cores even though LinearSVC itself does not expose
    # an n_jobs argument in this sklearn version.
    limits = None if n_jobs in (-1, None) else n_jobs

    linear_kwargs = dict(
        C=C,
        class_weight=class_weight,
        dual=True,
        loss="squared_hinge",
        tol=tol,
        max_iter=max_iter,
        fit_intercept=True,
        random_state=42,
    )
    linear_kwargs.update(kwargs)
    linear_kwargs.pop("n_jobs", None)

    with threadpool_limits(limits=limits):
        model = LinearSVC(**linear_kwargs)
        model.fit(X_train, y_train)
    return model
