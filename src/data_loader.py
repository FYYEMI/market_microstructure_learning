"""Raw order book loading and standardization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import infer_symbol_frequency_from_filename, parse_timestamp_series


def _read_raw_file(file_path: Path, nrows: int | None = None) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path, nrows=nrows, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(file_path)
        if nrows is not None:
            return df.head(nrows).copy()
        return df
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def _first_scalar(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, pd.Series):
        non_null = value.dropna()
        return non_null.iloc[0] if not non_null.empty else None
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _find_timestamp_column(columns: list[str]) -> str | None:
    preferred = [
        "timestamp",
        "system_time",
        "datetime",
        "date",
        "time",
        "event_time",
    ]
    for column in preferred:
        if column in columns:
            return column
    return None


def load_raw_order_book(file_path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """
    Load a raw order book file, parse the timestamp, sort chronologically, and
    drop duplicate timestamps.
    """

    path = Path(file_path)
    df = _read_raw_file(path, nrows=nrows)
    timestamp_col = _find_timestamp_column(list(df.columns))
    if timestamp_col is not None:
        df = df.copy()
        df["timestamp"] = parse_timestamp_series(df[timestamp_col])
    elif "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = parse_timestamp_series(df["timestamp"])
    else:
        raise ValueError("No timestamp column found in raw file.")

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    symbol, frequency = infer_symbol_frequency_from_filename(path)
    df["symbol"] = symbol
    df["frequency"] = frequency
    df["source_file"] = path.name
    return df


def inspect_raw_files(raw_dir: str | Path, sample_rows: int = 3) -> list[dict[str, Any]]:
    """
    Print a compact inventory of raw files and return a summary list.
    """

    raw_path = Path(raw_dir)
    summaries: list[dict[str, Any]] = []
    files = sorted([p for p in raw_path.iterdir() if p.is_file()])
    for file_path in files:
        summary: dict[str, Any] = {
            "file_name": file_path.name,
            "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
        }
        try:
            sample_df = _read_raw_file(file_path, nrows=sample_rows)
            summary["sample_shape"] = sample_df.shape
            summary["columns"] = list(sample_df.columns)
            summaries.append(summary)
            print(f"File: {file_path.name}")
            print(f"  Size MB: {summary['file_size_mb']}")
            print(f"  Sample shape: {sample_df.shape}")
            print(f"  Columns: {list(sample_df.columns)}")
            print(sample_df.head(sample_rows).to_string())
            print("-" * 80)
        except Exception as exc:  # pragma: no cover - inspection path
            summary["error"] = str(exc)
            summaries.append(summary)
            print(f"File: {file_path.name}")
            print(f"  Error: {exc}")
            print("-" * 80)
    return summaries


def _standardize_level_columns(
    df: pd.DataFrame,
    standardized: pd.DataFrame,
    level: int,
    midpoint: pd.Series | None,
) -> None:
    source_index = level - 1
    bid_p_col = f"bid_p{level}"
    ask_p_col = f"ask_p{level}"
    bid_q_col = f"bid_q{level}"
    ask_q_col = f"ask_q{level}"

    if bid_p_col in df.columns and ask_p_col in df.columns:
        standardized[bid_p_col] = pd.to_numeric(df[bid_p_col], errors="coerce")
        standardized[ask_p_col] = pd.to_numeric(df[ask_p_col], errors="coerce")
    else:
        bid_distance_col = f"bids_distance_{source_index}"
        ask_distance_col = f"asks_distance_{source_index}"
        if midpoint is not None and bid_distance_col in df.columns and ask_distance_col in df.columns:
            bid_distance = pd.to_numeric(df[bid_distance_col], errors="coerce")
            ask_distance = pd.to_numeric(df[ask_distance_col], errors="coerce")
            standardized[bid_p_col] = midpoint * (1 + bid_distance)
            standardized[ask_p_col] = midpoint * (1 + ask_distance)
        elif midpoint is not None and level == 1 and "spread" in df.columns:
            spread = pd.to_numeric(df["spread"], errors="coerce")
            standardized[bid_p_col] = midpoint - spread / 2
            standardized[ask_p_col] = midpoint + spread / 2
        else:
            standardized[bid_p_col] = np.nan
            standardized[ask_p_col] = np.nan

    if bid_q_col in df.columns and ask_q_col in df.columns:
        standardized[bid_q_col] = pd.to_numeric(df[bid_q_col], errors="coerce")
        standardized[ask_q_col] = pd.to_numeric(df[ask_q_col], errors="coerce")
    else:
        bid_notional_col = f"bids_notional_{source_index}"
        ask_notional_col = f"asks_notional_{source_index}"
        if bid_notional_col in df.columns and ask_notional_col in df.columns:
            standardized[bid_q_col] = pd.to_numeric(df[bid_notional_col], errors="coerce")
            standardized[ask_q_col] = pd.to_numeric(df[ask_notional_col], errors="coerce")
        else:
            standardized[bid_q_col] = np.nan
            standardized[ask_q_col] = np.nan


def standardize_lob_columns(df: pd.DataFrame, source_file: str | None = None) -> pd.DataFrame:
    """
    Convert the raw Kaggle schema into the internal standardized order book schema.
    """

    standardized = pd.DataFrame(index=df.index)
    source_name = _first_scalar(source_file if source_file is not None else df.get("source_file"))
    symbol, frequency = infer_symbol_frequency_from_filename(source_name)

    if "timestamp" in df.columns:
        standardized["timestamp"] = parse_timestamp_series(df["timestamp"])
    elif "system_time" in df.columns:
        standardized["timestamp"] = parse_timestamp_series(df["system_time"])
    else:
        raise ValueError("A timestamp column is required for standardization.")

    standardized["symbol"] = symbol if symbol is not None else _first_scalar(df.get("symbol"))
    standardized["frequency"] = frequency if frequency is not None else _first_scalar(df.get("frequency"))
    standardized["source_file"] = Path(source_name).name if source_name is not None else _first_scalar(df.get("source_file"))

    midpoint = pd.to_numeric(df["midpoint"], errors="coerce") if "midpoint" in df.columns else None
    for level in range(1, 16):
        _standardize_level_columns(df, standardized, level, midpoint)

    standardized = standardized.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    return standardized
