"""Rule-based signal-adjusted execution schedules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .execution import ParentOrder, select_execution_window


def _prepare_market_df(market_df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in market_df.columns:
        raise ValueError("market_df must contain a timestamp column.")
    prepared = market_df.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True, errors="coerce")
    prepared = prepared.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if prepared.empty:
        raise ValueError("market_df is empty after timestamp parsing.")
    return prepared


def _select_rows_evenly(window_df: pd.DataFrame, num_slices: int) -> pd.DataFrame:
    if window_df.empty:
        raise ValueError("The execution window is empty.")
    if num_slices <= 0:
        raise ValueError("num_slices must be positive.")
    positions = np.linspace(0, len(window_df) - 1, num=num_slices)
    row_idx = np.clip(np.round(positions).astype(int), 0, len(window_df) - 1)
    return window_df.iloc[row_idx].reset_index(drop=True)


def _normalize_series_to_target(values: pd.Series, target_qty: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    total = float(numeric.sum())
    if total <= 0:
        return pd.Series(np.repeat(target_qty / len(numeric), len(numeric)), index=numeric.index, dtype=float)
    return numeric * (target_qty / total)


def generate_obi_adjusted_twap_schedule(
    order: ParentOrder,
    market_df: pd.DataFrame,
    obi_col: str = "obi_5",
    spread_col: str = "relative_spread",
    depth_col: str = "total_depth_5",
    obi_up_threshold: float = 0.2,
    obi_down_threshold: float = -0.2,
    accelerate_multiplier: float = 1.5,
    slowdown_multiplier: float = 0.5,
    use_regime_filter: bool = True,
    spread_threshold: float | None = None,
    depth_threshold: float | None = None,
) -> pd.DataFrame:
    """Create a schedule that speeds up or slows down TWAP using a simple OBI rule.

    The raw child quantities are normalized back to the target quantity so that
    total executed quantity remains comparable to static TWAP.
    """

    prepared = _prepare_market_df(market_df)
    window_df = select_execution_window(prepared, order.start_time, order.end_time)
    selected = _select_rows_evenly(window_df, order.num_slices)

    missing = [column for column in [obi_col, spread_col, depth_col] if column not in selected.columns]
    if missing:
        raise ValueError(f"market_df is missing required columns: {missing}")

    base_child_qty = float(order.target_qty) / float(order.num_slices)
    selected_spread = pd.to_numeric(selected[spread_col], errors="coerce")
    selected_depth = pd.to_numeric(selected[depth_col], errors="coerce")

    raw_child_qty: list[float] = []
    child_qty: list[float] = []
    base_signal: list[str] = []
    regime_pass: list[bool] = []
    multipliers: list[float] = []
    remaining_qty = float(order.target_qty)

    for row_idx, row in selected.iterrows():
        obi_value = pd.to_numeric(pd.Series([row[obi_col]]), errors="coerce").iloc[0]
        spread_value = pd.to_numeric(pd.Series([row[spread_col]]), errors="coerce").iloc[0]
        depth_value = pd.to_numeric(pd.Series([row[depth_col]]), errors="coerce").iloc[0]
        if spread_threshold is None:
            observed_spreads = selected_spread.iloc[: len(raw_child_qty) + 1].dropna()
            spread_cutoff = float(observed_spreads.median()) if not observed_spreads.empty else np.nan
        else:
            spread_cutoff = float(spread_threshold)
        if depth_threshold is None:
            observed_depth = selected_depth.iloc[: len(raw_child_qty) + 1].dropna()
            depth_cutoff = float(observed_depth.median()) if not observed_depth.empty else np.nan
        else:
            depth_cutoff = float(depth_threshold)

        if use_regime_filter:
            pass_regime = (
                pd.notna(obi_value)
                and pd.notna(spread_value)
                and pd.notna(depth_value)
                and spread_value <= spread_cutoff
                and depth_value >= depth_cutoff
                and abs(float(obi_value)) >= 0.2
            )
        else:
            pass_regime = True
        regime_pass.append(bool(pass_regime))

        if use_regime_filter and not pass_regime:
            multiplier = 1.0
            signal = "blocked_by_regime_filter"
        elif float(obi_value) > obi_up_threshold:
            multiplier = accelerate_multiplier
            signal = "accelerate"
        elif float(obi_value) < obi_down_threshold:
            multiplier = slowdown_multiplier
            signal = "slowdown"
        else:
            multiplier = 1.0
            signal = "neutral"

        multipliers.append(float(multiplier))
        base_signal.append(signal)
        raw_qty = base_child_qty * float(multiplier)
        raw_child_qty.append(raw_qty)
        remaining_slices = len(selected) - int(row_idx)
        if remaining_slices <= 1:
            next_child_qty = remaining_qty
        else:
            next_child_qty = min(remaining_qty, (remaining_qty / remaining_slices) * float(multiplier))
        child_qty.append(float(next_child_qty))
        remaining_qty = max(remaining_qty - float(next_child_qty), 0.0)

    schedule = pd.DataFrame(
        {
            "timestamp": selected["timestamp"].values,
            "strategy": "obi_adjusted_twap",
            "base_child_qty": np.repeat(base_child_qty, len(selected)),
            "raw_child_qty": raw_child_qty,
            "obi_5": selected[obi_col].values if obi_col in selected.columns else np.nan,
            "relative_spread": selected[spread_col].values if spread_col in selected.columns else np.nan,
            "total_depth_5": selected[depth_col].values if depth_col in selected.columns else np.nan,
            "adjustment_signal": base_signal,
            "regime_filter_pass": regime_pass,
            "adjustment_multiplier": multipliers,
        }
    )
    schedule["child_qty"] = child_qty
    return schedule[
        [
            "timestamp",
            "strategy",
            "base_child_qty",
            "raw_child_qty",
            "child_qty",
            "obi_5",
            "relative_spread",
            "total_depth_5",
            "adjustment_signal",
            "regime_filter_pass",
            "adjustment_multiplier",
        ]
    ].reset_index(drop=True)
