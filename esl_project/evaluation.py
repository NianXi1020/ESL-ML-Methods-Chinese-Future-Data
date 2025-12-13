from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
)
from sklearn.preprocessing import StandardScaler

from .data_utils import get_feature_matrix, rolling_month_windows
from .model_loader import ModelConfig, train_model


def compute_class_weights(y: Iterable[int]) -> Dict[int, float]:
    counts = pd.Series(list(y)).value_counts().to_dict()
    if not counts:
        return {}
    inv = {cls: 1.0 / cnt for cls, cnt in counts.items()}
    total = sum(inv.values())
    return {cls: val / total for cls, val in inv.items()}


def evaluate_model_grid(
    df: pd.DataFrame,
    model_name: str,
    param_grid: List[Dict[str, object]],
    feature_list: List[str],
    train_months: int = 12,
    test_months: int = 1,
    metric_dir: Optional[Path] = None,
    tag: Optional[str] = None,
) -> Dict[str, object]:
    """Evaluate a hyperparameter grid with rolling windows and return the best result."""

    grid_rows: List[Dict[str, float]] = []
    per_result: Dict[int, Dict[str, object]] = {}

    for idx, params in enumerate(param_grid):
        fold_metrics: List[Dict[str, float]] = []
        preds: List[pd.DataFrame] = []
        conf_mats: List[np.ndarray] = []

        for fold_idx, (train_df, test_df, train_months_list, test_months_list) in enumerate(
            rolling_month_windows(df, train_months=train_months, test_months=test_months), start=1
        ):
            X_train_df = get_feature_matrix(train_df, feature_list)
            X_test_df = get_feature_matrix(test_df, feature_list)
            y_train_orig = train_df["label"].astype(int)
            y_test_orig = test_df["label"].astype(int)

            # XGBoost expects labels starting at 0..K-1; map for training and map back for metrics
            if model_name == "xgb":
                encode_map = {-1: 0, 0: 1, 1: 2}
                decode_map = {v: k for k, v in encode_map.items()}
                y_train = y_train_orig.map(encode_map)
                y_test = y_test_orig.map(encode_map)
            else:
                y_train = y_train_orig
                y_test = y_test_orig

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_df)
            X_test = scaler.transform(X_test_df)

            class_weights = compute_class_weights(y_train_orig)
            merged_params = dict(params)
            if model_name != "xgb":
                merged_params.setdefault("class_weight", class_weights)
            else:
                merged_params.pop("class_weight", None)

            config = ModelConfig(model_name=model_name, params=merged_params)
            model = train_model(X_train, y_train, config)

            y_pred_enc = model.predict(X_test)

            # Map predictions/probabilities back to original labels if needed
            prob = np.zeros((len(y_pred_enc), 3))
            score_one = None
            if model_name == "xgb":
                y_pred = pd.Series(y_pred_enc).map(decode_map).values
                if hasattr(model, "predict_proba"):
                    prob_enc = model.predict_proba(X_test)
                    for col_idx, orig_label in enumerate([-1, 0, 1]):
                        enc_idx = encode_map[orig_label]
                        if enc_idx < prob_enc.shape[1]:
                            prob[:, col_idx] = prob_enc[:, enc_idx]
            else:
                y_pred = y_pred_enc
                if hasattr(model, "predict_proba"):
                    prob_raw = model.predict_proba(X_test)
                    # Align probability columns to [-1, 0, 1] if possible
                    if hasattr(model, "classes_"):
                        cls_list = list(model.classes_)
                        for idx_cls, lbl in enumerate(cls_list):
                            if lbl in [-1, 0, 1]:
                                target_idx = [-1, 0, 1].index(lbl)
                                if target_idx < prob_raw.shape[1]:
                                    prob[:, target_idx] = prob_raw[:, idx_cls]
                    else:
                        prob[:, : prob_raw.shape[1]] = prob_raw
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(X_test)
                    if scores.ndim == 1:
                        score_one = scores
                    else:
                        # Map decision scores to class +1 if present
                        if hasattr(model, "classes_") and 1 in model.classes_:
                            score_one = scores[:, list(model.classes_).index(1)]
                        else:
                            score_one = scores[:, 0]
                    # convert decision scores to pseudo probabilities for class 1
                    if score_one is not None:
                        prob[:, 2] = 1.0 / (1.0 + np.exp(-score_one))

            acc = accuracy_score(y_test_orig, y_pred)
            f1 = f1_score(y_test_orig, y_pred, average="macro", zero_division=0)
            prec_pos = precision_score(y_test_orig, y_pred, labels=[1], average="macro", zero_division=0)
            prec_neg = precision_score(y_test_orig, y_pred, labels=[-1], average="macro", zero_division=0)

            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "train_start": train_months_list[0],
                    "train_end": train_months_list[-1],
                    "test_start": test_months_list[0],
                    "test_end": test_months_list[-1],
                    "accuracy": acc,
                    "f1_macro": f1,
                    "precision_pos": prec_pos,
                    "precision_neg": prec_neg,
                }
            )

            conf = confusion_matrix(y_test_orig, y_pred, labels=[-1, 0, 1])
            conf_mats.append(conf)

            preds.append(
                pd.DataFrame(
                    {
                        "index": test_df["index"],
                        "true": y_test.values,
                        "pred": y_pred,
                        "prob_-1": prob[:, 0] if prob.shape[1] > 0 else np.nan,
                        "prob_0": prob[:, 1] if prob.shape[1] > 1 else np.nan,
                        "prob_1": prob[:, 2] if prob.shape[1] > 2 else np.nan,
                        "score_1": score_one if score_one is not None else np.nan,
                    }
                )
            )

        metrics_df = pd.DataFrame(fold_metrics)
        avg_row = {
            "param_index": idx,
            "avg_accuracy": metrics_df["accuracy"].mean(),
            "avg_f1_macro": metrics_df["f1_macro"].mean(),
            "avg_precision_pos": metrics_df["precision_pos"].mean(),
            "avg_precision_neg": metrics_df["precision_neg"].mean(),
        }
        grid_rows.append(avg_row)

        per_result[idx] = {
            "metrics": metrics_df,
            "predictions": pd.concat(preds, ignore_index=True),
            "confusions": conf_mats,
        }

    summary_df = pd.DataFrame(grid_rows)
    summary_df = summary_df.sort_values(["avg_f1_macro", "avg_accuracy"], ascending=False)
    best_idx = int(summary_df.iloc[0]["param_index"])

    if metric_dir is not None:
        metric_dir.mkdir(parents=True, exist_ok=True)
        name_prefix = f"{tag}_{model_name}" if tag else model_name
        summary_df.to_csv(metric_dir / f"{name_prefix}_grid_summary.csv", index=False)

    chosen = per_result[best_idx]
    agg_conf = sum(chosen["confusions"])
    if metric_dir is not None:
        name_prefix = f"{tag}_{model_name}" if tag else model_name
        chosen["metrics"].to_csv(metric_dir / f"{name_prefix}_rolling_metrics.csv", index=False)
        chosen["predictions"].to_csv(metric_dir / f"{name_prefix}_predictions.csv", index=False)
        pd.DataFrame(agg_conf, index=[-1, 0, 1], columns=[-1, 0, 1]).to_csv(
            metric_dir / f"{name_prefix}_confusion.csv"
        )

    best_params = param_grid[best_idx]
    return {
        "best_params": best_params,
        "best_index": best_idx,
        "summary": summary_df,
        "rolling_metrics": chosen["metrics"],
        "predictions": chosen["predictions"],
        "confusion": agg_conf,
    }
