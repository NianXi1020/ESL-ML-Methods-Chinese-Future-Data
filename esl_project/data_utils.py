from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# Baseline feature set shared by the pipeline and notebooks
BASELINE_FEATURES: List[str] = [
    "r_t",
    "r_t_minus_1",
    "r_t_minus_2",
    "r_t_minus_3",
    "r_t_minus_4",
    "r_t_minus_5",
    "ma_5",
    "rsi_5",
    "range_5",
    "abs_r_t",
    "r_times_volume",
    "co_move",
]


@dataclass
class DataLoadConfig:
    data_dir: Path
    max_files: Optional[int] = None
    nrows_per_file: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# -----------------------------
# Data loading helpers
# -----------------------------

def list_csv_files(data_dir: Path, limit: Optional[int] = None) -> List[Path]:
    files = sorted(data_dir.glob("*.csv"))
    if limit is not None:
        files = files[:limit]
    return files


def load_single_csv(
    path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["index"], nrows=nrows)
    if start_date is not None:
        df = df[df["index"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["index"] <= pd.to_datetime(end_date)]
    return df.sort_values("index").reset_index(drop=True)


def load_dataset(cfg: DataLoadConfig) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    files = list_csv_files(cfg.data_dir, cfg.max_files)
    for path in files:
        frames.append(
            load_single_csv(
                path,
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                nrows=cfg.nrows_per_file,
            )
        )
    if not frames:
        raise ValueError("No CSV files were loaded; please check the data directory.")
    return pd.concat(frames, ignore_index=True)


# -----------------------------
# Feature engineering
# -----------------------------

def compute_return(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"]).diff()


def add_feature_columns(
    df: pd.DataFrame,
    lag: int = 5,
    rsi_window: int = 5,
    range_window: int = 5,
) -> pd.DataFrame:
    df = df.copy()

    df["r_t"] = compute_return(df)
    for k in range(1, lag + 1):
        df[f"r_t_minus_{k}"] = df["r_t"].shift(k)

    df[f"ma_{rsi_window}"] = df["close"].rolling(rsi_window).mean()

    delta_close = df["close"].diff()
    gain = delta_close.clip(lower=0)
    loss = (-delta_close).clip(lower=0)
    avg_gain = gain.rolling(rsi_window).mean()
    avg_loss = loss.rolling(rsi_window).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    df[f"rsi_{rsi_window}"] = rsi.fillna(100)

    df["range_1m"] = df["high"] - df["low"]
    rolling_high = df["high"].rolling(range_window).max()
    rolling_low = df["low"].rolling(range_window).min()
    df[f"range_{range_window}"] = rolling_high - rolling_low

    df["volume_change"] = df["volume"].diff()
    df["oi_change"] = df["open_interest"].diff()
    df["spread_vwap"] = df["close"] - df["avg"]

    df["abs_r_t"] = df["r_t"].abs()
    df["r_times_volume"] = df["r_t"] * df["volume"]
    df["co_move"] = df["close"] - df["open"]
    return df


# -----------------------------
# Label creation and alignment
# -----------------------------

def generate_labels(df: pd.DataFrame, alpha: float = 0.001) -> pd.Series:
    future_close = df["close"].shift(-1)
    ratio = future_close / df["close"]
    labels = pd.Series(index=df.index, dtype=float)
    labels[ratio > 1 + alpha] = 1
    labels[ratio < 1 - alpha] = -1
    labels[(ratio <= 1 + alpha) & (ratio >= 1 - alpha)] = 0
    return labels


def prepare_labeled_data(df: pd.DataFrame, alpha: float = 0.001) -> pd.DataFrame:
    df = df.copy()
    df["label"] = generate_labels(df, alpha=alpha)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    return df


def get_feature_matrix(df: pd.DataFrame, feature_list: Iterable[str]) -> pd.DataFrame:
    return df[list(feature_list)].copy()


# -----------------------------
# Rolling windows
# -----------------------------

def month_period(series: pd.Series) -> pd.Series:
    return series.dt.to_period("M")


def rolling_month_windows(
    df: pd.DataFrame,
    train_months: int = 12,
    test_months: int = 1,
) -> Iterable[Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]]:
    periods = month_period(df["index"])
    unique_months = pd.PeriodIndex(periods.unique()).sort_values()
    start = 0
    while start + train_months + test_months <= len(unique_months):
        train_slice = unique_months[start : start + train_months]
        test_slice = unique_months[start + train_months : start + train_months + test_months]

        train_df = df[periods.isin(train_slice)]
        test_df = df[periods.isin(test_slice)]

        yield (
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
            train_slice.astype(str).tolist(),
            test_slice.astype(str).tolist(),
        )
        start += 1
