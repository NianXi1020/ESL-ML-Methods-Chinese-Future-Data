from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_logit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    class_weight: Optional[dict[str, float]] = None,
    n_jobs: int = -1,
    max_iter: int = 200,
    solver: str = "lbfgs",
    multi_class: str = "multinomial",
    **kwargs: Any,
) -> LogisticRegression:
    """Train a multinomial logistic regression model."""
    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        n_jobs=n_jobs,
        max_iter=max_iter,
        solver=solver,
        multi_class=multi_class,
        penalty="l2",
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model
