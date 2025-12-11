from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .data_utils import (
    BASELINE_FEATURES,
    DataLoadConfig,
    add_feature_columns,
    downsample_time_series,
    list_csv_files,
    load_single_csv,
    prepare_labeled_data,
    select_features_with_l1,
)
from .evaluation import evaluate_model_grid


@dataclass
class ContractRunConfig:
    candidate_C: List[float]
    model_name: str = "logit"
    param_grid: Optional[List[Dict[str, object]]] = None
    train_months: int = 12
    test_months: int = 1
    alpha: float = 0.001
    downsample_every: int = 1
    select_features: bool = False
    l1_C: float = 0.1
    feature_list: List[str] = None  # type: ignore

    def __post_init__(self):
        if self.feature_list is None:
            self.feature_list = BASELINE_FEATURES
        if self.param_grid is None and self.model_name.lower() == "logit":
            self.param_grid = [{"C": c} for c in self.candidate_C]
        elif self.param_grid is None:
            self.param_grid = [{}]


def short_contract_tag(contract_name: str) -> str:
    token = contract_name.split("_")[0]
    return token[:2].upper()


def run_single_contract(
    contract_path: Path,
    output_root: Path,
    run_cfg: ContractRunConfig,
    load_cfg: DataLoadConfig,
) -> Dict[str, object]:
    contract_name = contract_path.stem
    tag = short_contract_tag(contract_name)

    contract_root = output_root / tag
    fig_dir = contract_root / "figures"
    metric_dir = contract_root / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_single_csv(
        contract_path,
        start_date=load_cfg.start_date,
        end_date=load_cfg.end_date,
        nrows=load_cfg.nrows_per_file,
    )
    df_feat = add_feature_columns(df_raw)
    df_feat = downsample_time_series(df_feat, step=run_cfg.downsample_every)
    df_labeled = prepare_labeled_data(df_feat, alpha=run_cfg.alpha)

    # Drop any rows that still contain NaNs in the active feature set or label
    cols_to_check = list(run_cfg.feature_list) + ["label"]
    df_labeled = df_labeled.dropna(subset=cols_to_check).reset_index(drop=True)

    selected_features = run_cfg.feature_list
    if run_cfg.select_features and not df_labeled.empty:
        selected_features = select_features_with_l1(
            df_labeled,
            feature_list=run_cfg.feature_list,
            C=run_cfg.l1_C,
        )
        if not selected_features:
            selected_features = run_cfg.feature_list

    if df_labeled.empty:
        return {
            "contract": contract_name,
            "tag": tag,
            "best_params": None,
            "metrics": None,
            "note": "No labeled data after preprocessing",
        }

    results = evaluate_model_grid(
        df=df_labeled,
        model_name=run_cfg.model_name,
        param_grid=run_cfg.param_grid,
        feature_list=selected_features,
        train_months=run_cfg.train_months,
        test_months=run_cfg.test_months,
        metric_dir=metric_dir,
        tag=tag,
    )

    params_path = metric_dir / f"{tag}_{run_cfg.model_name}_params.txt"
    params_path.write_text(
        "Model: {model}\nBest params: {params}\nFeatures: {features}\nTrain months: {train_m} Test months: {test_m}\n".format(
            model=run_cfg.model_name,
            params=results["best_params"],
            features=selected_features,
            train_m=run_cfg.train_months,
            test_m=run_cfg.test_months,
        )
    )

    return {
        "contract": contract_name,
        "tag": tag,
        "best_params": results["best_params"],
        "summary": results["summary"],
        "rolling_metrics": results["rolling_metrics"],
        "predictions": results["predictions"],
        "confusion": results["confusion"],
        "fig_dir": fig_dir,
        "metric_dir": metric_dir,
        "features_used": selected_features,
    }


def build_tasks_from_dir(
    data_dir: Path,
    output_root: Path,
    run_cfg: ContractRunConfig,
    load_cfg: DataLoadConfig,
    max_files: Optional[int] = None,
) -> List[tuple]:
    files = list_csv_files(data_dir, limit=max_files)
    tasks = []
    for path in files:
        tasks.append((path, output_root, run_cfg, load_cfg))
    return tasks
