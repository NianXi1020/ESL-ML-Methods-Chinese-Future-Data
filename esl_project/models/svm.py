from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.svm import SVC


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    gamma: str | float = "scale",
    class_weight: Optional[dict[str, float]] = None,
    probability: bool = True,
    **kwargs: Any,
) -> SVC:
    """Train an RBF SVM classifier with optional class weights."""
    model = SVC(
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        probability=probability,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model
