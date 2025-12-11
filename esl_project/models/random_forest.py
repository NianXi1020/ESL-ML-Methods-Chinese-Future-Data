from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    max_features: str | int | float | None = "sqrt",
    class_weight: Optional[dict[str, float]] = None,
    n_jobs: int = -1,
    random_state: Optional[int] = 42,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Train a random forest classifier with sensible defaults for tabular data."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model
