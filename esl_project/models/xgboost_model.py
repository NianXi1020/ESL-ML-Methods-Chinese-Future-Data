from __future__ import annotations

from typing import Any, Optional

import numpy as np


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    n_jobs: int = -1,
    **kwargs: Any,
):
    """Train an XGBoost classifier (requires xgboost to be installed)."""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        n_jobs=n_jobs,
        objective="multi:softprob",
        eval_metric="mlogloss",
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model
