from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .models.logistic import train_logit
from .models.random_forest import train_random_forest
from .models.svm import train_svm
from .models.xgboost_model import train_xgboost


@dataclass
class ModelConfig:
    model_name: str
    params: Dict[str, Any]


def train_model(X_train: np.ndarray, y_train: np.ndarray, config: ModelConfig):
    """Dispatch to the requested model implementation."""
    name = config.model_name.lower()
    if name == "logit":
        return train_logit(X_train, y_train, **config.params)
    if name == "svm":
        return train_svm(X_train, y_train, **config.params)
    if name == "rf":
        return train_random_forest(X_train, y_train, **config.params)
    if name == "xgb":
        return train_xgboost(X_train, y_train, **config.params)
    raise ValueError(f"Unknown model name: {config.model_name}")
