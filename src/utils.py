"""Shared helper functions for the first milestone."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_symbol_frequency_from_filename(file_path: str | Path | None) -> tuple[str | None, str | None]:
    if file_path is None:
        return None, None
    stem = Path(file_path).stem
    parts = stem.split("_")
    if not parts:
        return None, None
    symbol = parts[0].upper()
    frequency = parts[1] if len(parts) > 1 else None
    return symbol, frequency


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series:
    result = numerator / denominator
    if isinstance(result, pd.Series):
        return result.where(pd.Series(denominator, index=result.index) != 0, np.nan)
    denominator_array = np.asarray(denominator)
    return pd.Series(np.where(denominator_array == 0, np.nan, result))


def top_level_columns(prefix: str, levels: int) -> list[str]:
    return [f"{prefix}{level}" for level in range(1, levels + 1)]


def core_top5_columns(levels: int = 5) -> list[str]:
    columns: list[str] = []
    for level in range(1, levels + 1):
        columns.extend([f"bid_p{level}", f"ask_p{level}", f"bid_q{level}", f"ask_q{level}"])
    return columns

