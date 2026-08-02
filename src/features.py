"""Microstructure feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import safe_divide


def add_basic_lob_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["mid_price"] = (result["bid_p1"] + result["ask_p1"]) / 2
    result["spread"] = result["ask_p1"] - result["bid_p1"]
    result["relative_spread"] = safe_divide(result["spread"], result["mid_price"])
    return result


def add_depth_features(df: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
    result = df.copy()
    bid_q_cols = [f"bid_q{level}" for level in range(1, levels + 1) if f"bid_q{level}" in result.columns]
    ask_q_cols = [f"ask_q{level}" for level in range(1, levels + 1) if f"ask_q{level}" in result.columns]
    bid_depth = result[bid_q_cols].sum(axis=1) if bid_q_cols else pd.Series(np.nan, index=result.index)
    ask_depth = result[ask_q_cols].sum(axis=1) if ask_q_cols else pd.Series(np.nan, index=result.index)
    result[f"bid_depth_{levels}"] = bid_depth
    result[f"ask_depth_{levels}"] = ask_depth
    result[f"total_depth_{levels}"] = bid_depth + ask_depth
    result["obi_1"] = safe_divide(result["bid_q1"] - result["ask_q1"], result["bid_q1"] + result["ask_q1"])
    result[f"obi_{levels}"] = safe_divide(bid_depth - ask_depth, bid_depth + ask_depth)
    return result


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "mid_price" not in result.columns:
        raise ValueError("mid_price must exist before adding return features.")
    result["mid_return_1s"] = result["mid_price"].pct_change(periods=1)
    result["mid_return_5s"] = result["mid_price"].pct_change(periods=5)
    result["mid_return_10s"] = result["mid_price"].pct_change(periods=10)
    return result
