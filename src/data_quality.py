"""Basic data quality checks for standardized LOB data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import core_top5_columns


def _timestamp_diffs_seconds(df: pd.DataFrame) -> pd.Series:
    if "timestamp" not in df.columns:
        return pd.Series(dtype="float64")
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").sort_values()
    return timestamps.diff().dt.total_seconds()


def check_timestamp_order(df: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    is_sorted = timestamps.is_monotonic_increasing
    return {
        "timestamp_sorted": bool(is_sorted),
        "timestamp_min": timestamps.min(),
        "timestamp_max": timestamps.max(),
    }


def check_no_duplicate_timestamps(df: pd.DataFrame) -> dict[str, Any]:
    duplicate_count = int(df["timestamp"].duplicated().sum()) if "timestamp" in df.columns else 0
    return {
        "duplicate_timestamp_count": duplicate_count,
        "duplicate_timestamp_ratio": duplicate_count / len(df) if len(df) else np.nan,
    }


def check_best_bid_less_than_best_ask(df: pd.DataFrame) -> dict[str, Any]:
    if not {"bid_p1", "ask_p1"}.issubset(df.columns):
        return {"bid_p1_lt_ask_p1_ratio": np.nan, "violations": np.nan}
    valid = df["bid_p1"] < df["ask_p1"]
    return {
        "bid_p1_lt_ask_p1_ratio": float(valid.mean()),
        "violations": int((~valid).sum()),
    }


def check_positive_prices_and_sizes(df: pd.DataFrame, levels: int = 5) -> dict[str, Any]:
    price_cols = [col for level in range(1, levels + 1) for col in (f"bid_p{level}", f"ask_p{level}")]
    size_cols = [col for level in range(1, levels + 1) for col in (f"bid_q{level}", f"ask_q{level}")]
    price_cols = [col for col in price_cols if col in df.columns]
    size_cols = [col for col in size_cols if col in df.columns]

    positive_price_ratio = df[price_cols].gt(0).stack().mean() if price_cols else np.nan
    non_negative_size_ratio = df[size_cols].ge(0).stack().mean() if size_cols else np.nan
    return {
        "positive_price_ratio_top_levels": float(positive_price_ratio) if pd.notna(positive_price_ratio) else np.nan,
        "non_negative_size_ratio_top_levels": float(non_negative_size_ratio) if pd.notna(non_negative_size_ratio) else np.nan,
    }


def check_spread_reasonable(df: pd.DataFrame) -> dict[str, Any]:
    if not {"bid_p1", "ask_p1"}.issubset(df.columns):
        return {"spread_min": np.nan, "spread_max": np.nan, "spread_mean": np.nan, "non_positive_spread_ratio": np.nan}
    spread = df["ask_p1"] - df["bid_p1"]
    return {
        "spread_min": float(spread.min()),
        "spread_max": float(spread.max()),
        "spread_mean": float(spread.mean()),
        "spread_median": float(spread.median()),
        "non_positive_spread_ratio": float((spread <= 0).mean()),
    }


def run_basic_quality_report(df: pd.DataFrame, levels: int = 5) -> dict[str, Any]:
    core_columns = [col for col in core_top5_columns(levels) if col in df.columns]
    timestamp_diff = _timestamp_diffs_seconds(df)
    report: dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_value_ratio_core_columns": df[core_columns].isna().mean().to_dict() if core_columns else {},
    }
    report.update(check_timestamp_order(df))
    report.update(check_no_duplicate_timestamps(df))
    report.update(check_best_bid_less_than_best_ask(df))
    report.update(check_positive_prices_and_sizes(df, levels=levels))
    report.update(check_spread_reasonable(df))
    if len(timestamp_diff.dropna()) > 0:
        report["timestamp_diff_seconds_summary"] = timestamp_diff.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict()
        report["percentage_of_1_second_gaps"] = float((timestamp_diff == 1).mean())
        report["maximum_timestamp_gap_seconds"] = float(timestamp_diff.max())
        report["number_of_gaps_larger_than_1_second"] = int((timestamp_diff > 1).sum())
    else:
        report["timestamp_diff_seconds_summary"] = {}
        report["percentage_of_1_second_gaps"] = np.nan
        report["maximum_timestamp_gap_seconds"] = np.nan
        report["number_of_gaps_larger_than_1_second"] = np.nan

    print("Basic quality report")
    print(f"  row count: {report['row_count']}")
    print(f"  column count: {report['column_count']}")
    print(f"  timestamp min: {report['timestamp_min']}")
    print(f"  timestamp max: {report['timestamp_max']}")
    print(f"  duplicate timestamp count: {report['duplicate_timestamp_count']}")
    print(f"  bid_p1 < ask_p1 ratio: {report['bid_p1_lt_ask_p1_ratio']}")
    print(f"  positive price ratio top levels: {report['positive_price_ratio_top_levels']}")
    print(f"  non-negative size ratio top levels: {report['non_negative_size_ratio_top_levels']}")
    print(f"  spread summary: {{'min': {report['spread_min']}, 'max': {report['spread_max']}, 'mean': {report['spread_mean']}, 'median': {report['spread_median']}}}")
    print(f"  percentage of 1-second gaps: {report['percentage_of_1_second_gaps']}")
    print(f"  maximum timestamp gap seconds: {report['maximum_timestamp_gap_seconds']}")
    print(f"  number of gaps larger than 1 second: {report['number_of_gaps_larger_than_1_second']}")
    print(f"  missing value ratio core columns: {report['missing_value_ratio_core_columns']}")
    return report
