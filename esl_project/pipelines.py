from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .data_utils import (
    BASELINE_FEATURES,
    DataLoadConfig,
    add_feature_columns,
    list_csv_files,
    load_single_csv,
    prepare_labeled_data,
)
from .evaluation import evaluate_c_grid


@dataclass
class ContractRunConfig:
    candidate_C: List[float]
    train_months: int = 12
    test_months: int = 1
    alpha: float = 0.001
    feature_list: List[str] = None  # type: ignore

    def __post_init__(self):
        if self.feature_list is None:
            self.feature_list = BASELINE_FEATURES


def short_contract_tag(contract_name: str) -> str:
    return contract_name[:2]


def run_single_contract(
    contract_path: Path,
    output_root: Path,
    run_cfg: ContractRunConfig,
    load_cfg: DataLoadConfig,
) -> Dict[str, object]:
    contract_name = contract_path.stem
    tag = short_contract_tag(contract_name)

    fig_dir = output_root / "figures" / contract_name
    metric_dir = output_root / "metrics" / contract_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_single_csv(
        contract_path,
        start_date=load_cfg.start_date,
        end_date=load_cfg.end_date,
        nrows=load_cfg.nrows_per_file,
    )
    df_feat = add_feature_columns(df_raw)
    df_labeled = prepare_labeled_data(df_feat, alpha=run_cfg.alpha)

    if df_labeled.empty:
        return {
            "contract": contract_name,
            "tag": tag,
            "best_C": None,
            "metrics": None,
            "note": "No labeled data after preprocessing",
        }

    results = evaluate_c_grid(
        df=df_labeled,
        C_values=run_cfg.candidate_C,
        feature_list=run_cfg.feature_list,
        train_months=run_cfg.train_months,
        test_months=run_cfg.test_months,
        metric_dir=metric_dir,
    )

    return {
        "contract": contract_name,
        "tag": tag,
        "best_C": results["best_C"],
        "summary": results["summary"],
        "rolling_metrics": results["rolling_metrics"],
        "predictions": results["predictions"],
        "confusion": results["confusion"],
        "fig_dir": fig_dir,
        "metric_dir": metric_dir,
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
