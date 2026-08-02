"""Simplified execution simulator for the MVP and depth-walking milestones."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _to_utc_timestamp(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def _validate_side(side: str) -> str:
    normalized = str(side).strip().lower()
    if normalized not in {"buy", "sell"}:
        raise ValueError("side must be either 'buy' or 'sell'.")
    return normalized


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


def _detect_real_volume_column(columns: Iterable[str]) -> str | None:
    candidates = [
        "trade_volume",
        "volume",
        "vol",
        "traded_volume",
        "matched_volume",
        "executed_volume",
    ]
    column_set = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in column_set:
            return column_set[candidate]
    return None


def _marketable_side_columns(side: str) -> tuple[list[str], list[str], str]:
    normalized = _validate_side(side)
    if normalized == "buy":
        price_cols = [f"ask_p{level}" for level in range(1, 6)]
        qty_cols = [f"ask_q{level}" for level in range(1, 6)]
        top_of_book_col = "ask_p1"
    else:
        price_cols = [f"bid_p{level}" for level in range(1, 6)]
        qty_cols = [f"bid_q{level}" for level in range(1, 6)]
        top_of_book_col = "bid_p1"
    return price_cols, qty_cols, top_of_book_col


def simulate_market_order_fill(
    row: pd.Series,
    side: str,
    quantity: float,
    levels: int = 5,
    allow_partial: bool = True,
) -> dict:
    """
    Simulate a marketable child order that can walk through visible depth.

    The first version only consumes the visible top `levels` of the book and
    optionally leaves the remainder unfilled if depth is insufficient.
    """

    normalized_side = _validate_side(side)
    requested_qty = float(quantity)
    if requested_qty < 0:
        raise ValueError("quantity must be non-negative.")
    if requested_qty == 0:
        return {
            "requested_qty": 0.0,
            "filled_qty": 0.0,
            "unfilled_qty": 0.0,
            "fill_rate": 1.0,
            "avg_fill_price": np.nan,
            "notional": 0.0,
            "levels_used": 0,
            "top_of_book_price": np.nan,
            "walked_beyond_top": False,
        }
    if levels <= 0:
        raise ValueError("levels must be positive.")

    price_cols, qty_cols, top_of_book_col = _marketable_side_columns(normalized_side)
    price_cols = price_cols[:levels]
    qty_cols = qty_cols[:levels]

    top_of_book_price = pd.to_numeric(pd.Series([row.get(top_of_book_col, np.nan)]), errors="coerce").iloc[0]
    remaining = requested_qty
    filled_qty = 0.0
    notional = 0.0
    levels_used = 0

    for price_col, qty_col in zip(price_cols, qty_cols):
        if remaining <= 0:
            break
        price = pd.to_numeric(pd.Series([row.get(price_col, np.nan)]), errors="coerce").iloc[0]
        depth = pd.to_numeric(pd.Series([row.get(qty_col, np.nan)]), errors="coerce").iloc[0]
        if pd.isna(price) or pd.isna(depth) or depth <= 0:
            continue
        fill_qty = min(remaining, float(depth))
        if fill_qty <= 0:
            continue
        filled_qty += fill_qty
        notional += fill_qty * float(price)
        remaining -= fill_qty
        levels_used += 1

    unfilled_qty = requested_qty - filled_qty
    if unfilled_qty > 0 and not allow_partial:
        raise ValueError(
            f"Insufficient visible depth to fill {requested_qty:.6f}; "
            f"filled {filled_qty:.6f} and unfilled {unfilled_qty:.6f}."
        )

    avg_fill_price = notional / filled_qty if filled_qty > 0 else np.nan
    fill_rate = filled_qty / requested_qty if requested_qty > 0 else np.nan

    return {
        "requested_qty": requested_qty,
        "filled_qty": filled_qty,
        "unfilled_qty": unfilled_qty,
        "fill_rate": fill_rate,
        "avg_fill_price": avg_fill_price,
        "notional": notional,
        "levels_used": levels_used,
        "top_of_book_price": top_of_book_price,
        "walked_beyond_top": bool(levels_used > 1),
    }


@dataclass
class ParentOrder:
    symbol: str
    side: str
    target_qty: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    num_slices: int

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol)
        self.side = _validate_side(self.side)
        self.target_qty = float(self.target_qty)
        self.num_slices = int(self.num_slices)
        self.start_time = _to_utc_timestamp(self.start_time)
        self.end_time = _to_utc_timestamp(self.end_time)
        if self.target_qty < 0:
            raise ValueError("target_qty must be non-negative.")
        if self.num_slices <= 0:
            raise ValueError("num_slices must be positive.")
        if self.end_time < self.start_time:
            raise ValueError("end_time must be greater than or equal to start_time.")


def select_execution_window(
    market_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    prepared = _prepare_market_df(market_df)
    start = _to_utc_timestamp(start_time)
    end = _to_utc_timestamp(end_time)
    window = prepared[(prepared["timestamp"] >= start) & (prepared["timestamp"] <= end)].copy()
    if window.empty:
        raise ValueError("The selected execution window is empty.")
    return window.sort_values("timestamp").reset_index(drop=True)


def generate_twap_schedule(order: ParentOrder, market_df: pd.DataFrame) -> pd.DataFrame:
    window = select_execution_window(market_df, order.start_time, order.end_time)
    selected = _select_rows_evenly(window, order.num_slices)
    child_qty = order.target_qty / order.num_slices
    schedule = pd.DataFrame(
        {
            "timestamp": selected["timestamp"].values,
            "child_qty": np.repeat(child_qty, len(selected)),
            "strategy": "twap",
            "weight": np.repeat(1.0, len(selected)),
        }
    )
    return schedule


def generate_immediate_schedule(order: ParentOrder, market_df: pd.DataFrame) -> pd.DataFrame:
    """Submit the full parent quantity at the first observable window snapshot."""

    window = select_execution_window(market_df, order.start_time, order.end_time)
    first = window.iloc[0]
    return pd.DataFrame(
        {
            "timestamp": [first["timestamp"]],
            "child_qty": [order.target_qty],
            "strategy": ["immediate_depth_walk"],
            "weight": [1.0],
        }
    )


def generate_front_loaded_schedule(order: ParentOrder, market_df: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic front-loaded schedule using only clock time."""

    window = select_execution_window(market_df, order.start_time, order.end_time)
    selected = _select_rows_evenly(window, order.num_slices)
    weights = np.linspace(order.num_slices, 1, num=order.num_slices, dtype=float)
    weights = weights / weights.sum() if weights.sum() else np.repeat(1 / order.num_slices, order.num_slices)
    return pd.DataFrame(
        {
            "timestamp": selected["timestamp"].values,
            "child_qty": order.target_qty * weights,
            "strategy": "front_loaded",
            "weight": weights,
        }
    )


def generate_back_loaded_schedule(order: ParentOrder, market_df: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic back-loaded schedule using only clock time."""

    window = select_execution_window(market_df, order.start_time, order.end_time)
    selected = _select_rows_evenly(window, order.num_slices)
    weights = np.linspace(1, order.num_slices, num=order.num_slices, dtype=float)
    weights = weights / weights.sum() if weights.sum() else np.repeat(1 / order.num_slices, order.num_slices)
    return pd.DataFrame(
        {
            "timestamp": selected["timestamp"].values,
            "child_qty": order.target_qty * weights,
            "strategy": "back_loaded",
            "weight": weights,
        }
    )


def generate_random_schedule(order: ParentOrder, market_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible random schedule independent of future market states."""

    window = select_execution_window(market_df, order.start_time, order.end_time)
    selected = _select_rows_evenly(window, order.num_slices)
    rng = np.random.default_rng(seed)
    weights = rng.random(order.num_slices)
    weights = weights / weights.sum() if weights.sum() else np.repeat(1 / order.num_slices, order.num_slices)
    return pd.DataFrame(
        {
            "timestamp": selected["timestamp"].values,
            "child_qty": order.target_qty * weights,
            "strategy": "random_fixed_seed",
            "weight": weights,
        }
    )


def generate_liquidity_weighted_schedule(
    order: ParentOrder,
    market_df: pd.DataFrame,
    volume_col: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Create a non-forward-looking liquidity-weighted schedule.

    This is a scheduling proxy, not a market VWAP reconstruction. Each child
    order uses the current observed volume or visible depth relative to the
    trailing median available at that decision point.
    """

    window = select_execution_window(market_df, order.start_time, order.end_time)
    selected = _select_rows_evenly(window, order.num_slices)

    detected_volume_col = volume_col if volume_col is not None and volume_col in window.columns else None
    if detected_volume_col is None:
        detected_volume_col = _detect_real_volume_column(window.columns)

    strategy_name = "volume_weighted_schedule_proxy" if detected_volume_col is not None else "liquidity_weighted_schedule"
    if detected_volume_col is not None:
        weights = pd.to_numeric(selected[detected_volume_col], errors="coerce")
    else:
        proxy_col = "ask_depth_5" if order.side == "buy" else "bid_depth_5"
        if proxy_col not in selected.columns:
            raise ValueError(f"{proxy_col} is required for the liquidity-weighted schedule.")
        weights = pd.to_numeric(selected[proxy_col], errors="coerce")

    observed_liquidity = weights.fillna(0.0).clip(lower=0.0).reset_index(drop=True)
    child_qty: list[float] = []
    liquidity_multiplier: list[float] = []
    remaining_qty = float(order.target_qty)
    for idx, current_liquidity in enumerate(observed_liquidity):
        remaining_slices = len(observed_liquidity) - idx
        if remaining_slices <= 1:
            child = remaining_qty
            multiplier = 1.0
        else:
            trailing = observed_liquidity.iloc[: idx + 1]
            trailing_median = float(trailing[trailing > 0].median()) if bool((trailing > 0).any()) else 0.0
            multiplier = 1.0 if trailing_median <= 0 else float(np.clip(float(current_liquidity) / trailing_median, 0.5, 1.5))
            child = min(remaining_qty, (remaining_qty / remaining_slices) * multiplier)
        child_qty.append(float(child))
        liquidity_multiplier.append(float(multiplier))
        remaining_qty = max(remaining_qty - float(child), 0.0)

    schedule = pd.DataFrame(
        {
            "timestamp": selected["timestamp"].values,
            "child_qty": child_qty,
            "strategy": strategy_name,
            "weight": observed_liquidity.values,
            "liquidity_multiplier": liquidity_multiplier,
        }
    )
    return schedule, strategy_name


def simulate_execution(
    order: ParentOrder,
    market_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    fill_method: str = "top_of_book",
    levels: int = 5,
) -> pd.DataFrame:
    prepared_market = _prepare_market_df(market_df)
    schedule = schedule_df.copy()
    if "timestamp" not in schedule.columns or "child_qty" not in schedule.columns:
        raise ValueError("schedule_df must contain timestamp and child_qty columns.")
    schedule["timestamp"] = pd.to_datetime(schedule["timestamp"], utc=True, errors="coerce")
    schedule = schedule.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if schedule.empty:
        raise ValueError("schedule_df is empty after timestamp parsing.")

    price_col = "ask_p1" if order.side == "buy" else "bid_p1"
    if price_col not in prepared_market.columns:
        raise ValueError(f"market_df must contain {price_col}.")

    market_slice = prepared_market.copy()
    merged = schedule.merge(market_slice, on="timestamp", how="left", suffixes=("", "_market"))
    if merged[price_col].isna().any():
        missing = int(merged[price_col].isna().sum())
        raise ValueError(f"{missing} scheduled timestamps could not be matched to market prices.")

    merged["side"] = order.side
    merged["fill_method"] = fill_method

    records: list[dict] = []
    fill_method_normalized = str(fill_method).strip().lower()
    if fill_method_normalized not in {"top_of_book", "depth_walk"}:
        raise ValueError("fill_method must be either 'top_of_book' or 'depth_walk'.")

    for _, row in merged.iterrows():
        child_qty = float(row["child_qty"])
        if child_qty < 0:
            raise ValueError("child_qty must be non-negative.")
        if fill_method_normalized == "top_of_book":
            execution_price = float(row[price_col])
            requested_qty = child_qty
            filled_qty = child_qty
            unfilled_qty = 0.0
            fill_rate = 1.0
            notional = child_qty * execution_price if child_qty > 0 else 0.0
            levels_used = 1 if child_qty > 0 else 0
            walked_beyond_top = False
        else:
            fill = simulate_market_order_fill(row, order.side, quantity=child_qty, levels=levels, allow_partial=True)
            requested_qty = float(fill["requested_qty"])
            filled_qty = float(fill["filled_qty"])
            unfilled_qty = float(fill["unfilled_qty"])
            fill_rate = float(fill["fill_rate"])
            execution_price = float(fill["avg_fill_price"]) if not np.isnan(fill["avg_fill_price"]) else np.nan
            notional = float(fill["notional"])
            levels_used = int(fill["levels_used"])
            walked_beyond_top = bool(fill["walked_beyond_top"])

        records.append(
            {
                "timestamp": row["timestamp"],
                "strategy": row["strategy"],
                "side": order.side,
                "child_qty": child_qty,
                "requested_qty": requested_qty,
                "filled_qty": filled_qty,
                "unfilled_qty": unfilled_qty,
                "fill_rate": fill_rate,
                "execution_price": execution_price,
                "notional": notional,
                "levels_used": levels_used,
                "walked_beyond_top": walked_beyond_top,
                "fill_method": fill_method_normalized,
            }
        )

    execution_df = pd.DataFrame(records)
    execution_df["cum_requested_qty"] = execution_df["requested_qty"].cumsum()
    execution_df["cum_filled_qty"] = execution_df["filled_qty"].cumsum()
    execution_df["cum_notional"] = execution_df["notional"].cumsum()
    execution_df["cum_qty"] = execution_df["cum_filled_qty"]
    execution_df["avg_execution_price_so_far"] = execution_df["cum_notional"] / execution_df["cum_filled_qty"].replace(0, np.nan)
    return execution_df[
        [
            "timestamp",
            "strategy",
            "side",
            "child_qty",
            "requested_qty",
            "filled_qty",
            "unfilled_qty",
            "fill_rate",
            "execution_price",
            "notional",
            "levels_used",
            "walked_beyond_top",
            "cum_requested_qty",
            "cum_filled_qty",
            "cum_qty",
            "cum_notional",
            "avg_execution_price_so_far",
            "fill_method",
        ]
    ].reset_index(drop=True)


def calculate_execution_metrics(
    order: ParentOrder,
    execution_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> dict:
    prepared_market = _prepare_market_df(market_df)
    if execution_df.empty:
        raise ValueError("execution_df is empty.")

    requested_qty = float(execution_df["requested_qty"].sum()) if "requested_qty" in execution_df.columns else float(execution_df["child_qty"].sum())
    executed_qty = float(execution_df["filled_qty"].sum()) if "filled_qty" in execution_df.columns else float(execution_df["child_qty"].sum())
    unfilled_qty = float(execution_df["unfilled_qty"].sum()) if "unfilled_qty" in execution_df.columns else max(requested_qty - executed_qty, 0.0)
    total_notional = float(execution_df["notional"].sum())
    avg_execution_price = total_notional / executed_qty if executed_qty > 0 else np.nan

    arrival_slice = prepared_market[prepared_market["timestamp"] >= order.start_time]
    if arrival_slice.empty:
        raise ValueError("Could not determine arrival price at or after the order start time.")
    arrival_price = float(arrival_slice.iloc[0]["mid_price"])

    if executed_qty > 0 and order.side == "buy":
        slippage = avg_execution_price - arrival_price
    elif executed_qty > 0:
        slippage = arrival_price - avg_execution_price
    else:
        slippage = np.nan

    slippage_bps = (slippage / arrival_price * 10000) if arrival_price and not np.isnan(arrival_price) else np.nan
    execution_cost = executed_qty * slippage if not np.isnan(slippage) else 0.0
    avg_levels_used = float(pd.to_numeric(execution_df["levels_used"], errors="coerce").mean()) if "levels_used" in execution_df.columns else np.nan
    pct_child_orders_walked_beyond_top = float(pd.to_numeric(execution_df["walked_beyond_top"], errors="coerce").fillna(0).mean()) if "walked_beyond_top" in execution_df.columns else np.nan
    side_depth_col = "ask_depth_5" if order.side == "buy" else "bid_depth_5"
    if side_depth_col in prepared_market.columns:
        avg_visible_depth = float(pd.to_numeric(prepared_market[side_depth_col], errors="coerce").mean())
    else:
        avg_visible_depth = np.nan
    order_size_to_visible_depth_ratio = order.target_qty / avg_visible_depth if avg_visible_depth and not np.isnan(avg_visible_depth) else np.nan

    return {
        "strategy": execution_df["strategy"].iloc[0],
        "side": order.side,
        "target_qty": order.target_qty,
        "requested_qty": requested_qty,
        "executed_qty": executed_qty,
        "unfilled_qty": unfilled_qty,
        "fill_rate": executed_qty / requested_qty if requested_qty else 1.0,
        "arrival_price": arrival_price,
        "avg_execution_price": avg_execution_price,
        "slippage": slippage,
        "slippage_bps": slippage_bps,
        "execution_cost": execution_cost,
        "remaining_unfilled_qty": unfilled_qty,
        "avg_visible_depth_for_side": avg_visible_depth,
        "order_size_to_visible_depth_ratio": order_size_to_visible_depth_ratio,
        "avg_levels_used": avg_levels_used,
        "pct_child_orders_walked_beyond_top": pct_child_orders_walked_beyond_top,
        "start_time": order.start_time,
        "end_time": order.end_time,
        "num_slices": order.num_slices,
        "fill_method": execution_df["fill_method"].iloc[0] if "fill_method" in execution_df.columns else "top_of_book",
    }


def add_penalized_execution_costs(metrics_df: pd.DataFrame, penalty_bps: Iterable[float]) -> pd.DataFrame:
    """
    Add penalty-adjusted execution cost sensitivity columns.

    penalty cost = unfilled quantity * arrival price * penalty bps / 10000.
    This is a transparent sensitivity assumption, not calibrated opportunity
    cost or full implementation shortfall.
    """

    result = metrics_df.copy()
    for penalty in penalty_bps:
        column = f"penalized_cost_{float(penalty):g}bps"
        result[column] = (
            pd.to_numeric(result["execution_cost"], errors="coerce").fillna(0.0)
            + pd.to_numeric(result["unfilled_qty"], errors="coerce").fillna(0.0)
            * pd.to_numeric(result["arrival_price"], errors="coerce").fillna(0.0)
            * float(penalty)
            / 10000.0
        )
    return result
