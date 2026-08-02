"""Future return label construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_future_return_label(
    df: pd.DataFrame,
    horizon_seconds: int = 30,
    threshold: float = 0.0001,
) -> pd.DataFrame:
    """
    This first version uses row-based shift for future 30-second labels.
    This is valid only when the data is regular 1-second snapshots.
    If timestamps are irregular or have many gaps, timestamp-based forward
    alignment should be implemented in a later version.
    """

    if "mid_price" not in df.columns:
        raise ValueError("mid_price must exist before constructing labels.")

    result = df.copy()
    future_mid_price_col = f"future_mid_price_{horizon_seconds}s"
    future_return_col = f"future_return_{horizon_seconds}s"

    result[future_mid_price_col] = result["mid_price"].shift(-horizon_seconds)
    result[future_return_col] = result[future_mid_price_col] / result["mid_price"] - 1

    result["label"] = np.select(
        [
            result[future_return_col] > threshold,
            result[future_return_col] < -threshold,
        ],
        ["Up", "Down"],
        default="Flat",
    )

    result = result.dropna(subset=[future_mid_price_col, future_return_col]).reset_index(drop=True)
    return result
