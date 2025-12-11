from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .models.logistic import train_logit


@dataclass
class ModelConfig:
    model_name: str
    params: Dict[str, Any]


def train_model(X_train: np.ndarray, y_train: np.ndarray, config: ModelConfig):
    """Dispatch to the requested model implementation."""
    name = config.model_name.lower()
    if name == "logit":
        return train_logit(X_train, y_train, **config.params)
    raise ValueError(f"Unknown model name: {config.model_name}")
