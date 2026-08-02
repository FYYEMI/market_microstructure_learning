"""Multi-window diagnosis for execution strategies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPORTS_DIR
from .execution import (
    ParentOrder,
    add_penalized_execution_costs,
    calculate_execution_metrics,
    generate_back_loaded_schedule,
    generate_front_loaded_schedule,
    generate_immediate_schedule,
    generate_random_schedule,
    generate_twap_schedule,
    generate_liquidity_weighted_schedule,
    select_execution_window,
    simulate_execution,
)
from .signal_execution import generate_obi_adjusted_twap_schedule
from .utils import ensure_directory


def _prepare_market_df(market_df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in market_df.columns:
        raise ValueError("market_df must contain a timestamp column.")
    prepared = market_df.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True, errors="coerce")
    prepared = prepared.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if prepared.empty:
        raise ValueError("market_df is empty after timestamp parsing.")
    return prepared


def _detect_real_volume_column(columns: list[str]) -> str | None:
    candidates = [
        "trade_volume",
        "volume",
        "vol",
        "traded_volume",
        "matched_volume",
        "executed_volume",
    ]
    lookup = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _arrival_price(window_df: pd.DataFrame, start_time: pd.Timestamp) -> float:
    after_start = window_df[window_df["timestamp"] >= start_time]
    if after_start.empty:
        raise ValueError("Cannot determine arrival price for the execution window.")
    return float(after_start.iloc[0]["mid_price"])


def _window_condition_summary(window_df: pd.DataFrame, start_time: pd.Timestamp, end_time: pd.Timestamp) -> dict:
    arrival_price = _arrival_price(window_df, start_time)
    end_slice = window_df[window_df["timestamp"] <= end_time]
    if end_slice.empty:
        raise ValueError("Cannot determine end price for the execution window.")
    end_mid_price = float(end_slice.iloc[-1]["mid_price"])
    returns = window_df["mid_price"].pct_change().dropna()
    return {
        "arrival_price": arrival_price,
        "end_mid_price": end_mid_price,
        "window_return_bps": (end_mid_price / arrival_price - 1) * 10000 if arrival_price else np.nan,
        "avg_spread": float(window_df["spread"].mean()) if "spread" in window_df.columns else np.nan,
        "avg_relative_spread": float(window_df["relative_spread"].mean()) if "relative_spread" in window_df.columns else np.nan,
        "avg_total_depth_5": float(window_df["total_depth_5"].mean()) if "total_depth_5" in window_df.columns else np.nan,
        "avg_abs_obi_5": float(window_df["obi_5"].abs().mean()) if "obi_5" in window_df.columns else np.nan,
        "mid_price_volatility": float(returns.std()) if not returns.empty else np.nan,
    }


def generate_execution_windows(
    market_df: pd.DataFrame,
    window_minutes: int = 30,
    step_minutes: int = 30,
    max_windows: int | None = 100,
    buffer_minutes: int = 5,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    prepared = _prepare_market_df(market_df)
    if window_minutes <= 0 or step_minutes <= 0:
        raise ValueError("window_minutes and step_minutes must be positive.")

    window_delta = pd.Timedelta(minutes=window_minutes)
    step_delta = pd.Timedelta(minutes=step_minutes)
    buffer_delta = pd.Timedelta(minutes=buffer_minutes)

    start_bound = prepared["timestamp"].iloc[0] + buffer_delta
    end_bound = prepared["timestamp"].iloc[-1] - buffer_delta

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start = start_bound
    while current_start + window_delta <= end_bound:
        current_end = current_start + window_delta
        window_df = prepared[(prepared["timestamp"] >= current_start) & (prepared["timestamp"] <= current_end)]
        if len(window_df) >= 2:
            windows.append((current_start, current_end))
        if max_windows is not None and len(windows) >= max_windows:
            break
        current_start = current_start + step_delta

    return windows


def run_execution_comparison_for_window(
    market_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    side: str = "buy",
    target_qty: float = 10.0,
    num_slices: int = 30,
    fill_method: str = "top_of_book",
    levels: int = 5,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    prepared = _prepare_market_df(market_df)
    window_df = select_execution_window(prepared, start_time, end_time)
    if len(window_df) < num_slices:
        raise ValueError("Execution window does not contain enough snapshots for the requested number of slices.")

    order = ParentOrder(
        symbol=str(window_df["symbol"].iloc[0]) if "symbol" in window_df.columns else "UNKNOWN",
        side=side,
        target_qty=target_qty,
        start_time=start_time,
        end_time=end_time,
        num_slices=num_slices,
    )

    twap_schedule = generate_twap_schedule(order, window_df)
    twap_execution = simulate_execution(order, window_df, twap_schedule, fill_method=fill_method, levels=levels)
    twap_metrics = calculate_execution_metrics(order, twap_execution, window_df)
    twap_metrics["strategy"] = "twap"

    volume_col = _detect_real_volume_column(list(window_df.columns))
    liquidity_schedule, strategy_name = generate_liquidity_weighted_schedule(order, window_df, volume_col=volume_col)
    liquidity_execution = simulate_execution(order, window_df, liquidity_schedule, fill_method=fill_method, levels=levels)
    liquidity_metrics = calculate_execution_metrics(order, liquidity_execution, window_df)
    liquidity_metrics["strategy"] = strategy_name

    twap_metrics.update(_window_condition_summary(window_df, start_time, end_time))
    liquidity_metrics.update(_window_condition_summary(window_df, start_time, end_time))

    metrics_df = pd.DataFrame([twap_metrics, liquidity_metrics])
    execution_details = {
        "twap": twap_execution,
        strategy_name: liquidity_execution,
    }
    return metrics_df, execution_details


def run_multi_window_execution_analysis(
    market_df: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    side: str = "buy",
    target_qty: float = 10.0,
    num_slices: int = 30,
    fill_method: str = "top_of_book",
    levels: int = 5,
) -> pd.DataFrame:
    prepared = _prepare_market_df(market_df)
    rows: list[dict] = []
    for window_id, (start_time, end_time) in enumerate(windows):
        metrics_df, _ = run_execution_comparison_for_window(
            prepared,
            start_time=start_time,
            end_time=end_time,
            side=side,
            target_qty=target_qty,
            num_slices=num_slices,
            fill_method=fill_method,
            levels=levels,
        )
        for _, row in metrics_df.iterrows():
            record = row.to_dict()
            record["window_id"] = window_id
            record["start_time"] = pd.to_datetime(start_time, utc=True)
            record["end_time"] = pd.to_datetime(end_time, utc=True)
            rows.append(record)
    return pd.DataFrame(rows)


def summarize_strategy_comparison(
    results_df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    required = {
        "window_id",
        "start_time",
        "end_time",
        "strategy",
        "side",
        "fill_method",
        "slippage_bps",
        "execution_cost",
        "fill_rate",
        "avg_levels_used",
        "pct_child_orders_walked_beyond_top",
        "window_return_bps",
        "avg_spread",
        "avg_relative_spread",
        "avg_total_depth_5",
        "avg_abs_obi_5",
        "mid_price_volatility",
    }
    missing = sorted(required - set(results_df.columns))
    if missing:
        raise ValueError(f"results_df is missing required columns: {missing}")

    pivot_slippage = results_df.pivot(index="window_id", columns="strategy", values="slippage_bps")
    pivot_cost = results_df.pivot(index="window_id", columns="strategy", values="execution_cost")
    meta_cols = [
        "window_id",
        "start_time",
        "end_time",
        "side",
        "fill_method",
        "window_return_bps",
        "avg_spread",
        "avg_relative_spread",
        "avg_total_depth_5",
        "avg_abs_obi_5",
        "mid_price_volatility",
    ]
    meta_df = results_df[meta_cols].drop_duplicates(subset=["window_id"]).set_index("window_id")

    summary = pd.DataFrame(index=meta_df.index)
    summary["start_time"] = meta_df["start_time"]
    summary["end_time"] = meta_df["end_time"]
    summary["side"] = meta_df["side"]
    summary["fill_method"] = meta_df["fill_method"]
    summary["twap_slippage_bps"] = pivot_slippage["twap"]
    proxy_col = next((col for col in pivot_slippage.columns if col != "twap"), None)
    if proxy_col is None:
        raise ValueError("Could not locate proxy strategy in results_df.")
    summary["proxy_slippage_bps"] = pivot_slippage[proxy_col]
    summary["proxy_minus_twap_slippage_bps"] = summary["proxy_slippage_bps"] - summary["twap_slippage_bps"]
    summary["twap_execution_cost"] = pivot_cost["twap"]
    summary["proxy_execution_cost"] = pivot_cost[proxy_col]
    summary["proxy_minus_twap_cost"] = summary["proxy_execution_cost"] - summary["twap_execution_cost"]
    pivot_fill_rate = results_df.pivot(index="window_id", columns="strategy", values="fill_rate")
    pivot_levels = results_df.pivot(index="window_id", columns="strategy", values="avg_levels_used")
    pivot_walked = results_df.pivot(index="window_id", columns="strategy", values="pct_child_orders_walked_beyond_top")
    summary["twap_fill_rate"] = pivot_fill_rate["twap"]
    summary["proxy_fill_rate"] = pivot_fill_rate[proxy_col]
    summary["twap_avg_levels_used"] = pivot_levels["twap"]
    summary["proxy_avg_levels_used"] = pivot_levels[proxy_col]
    summary["twap_pct_walked_beyond_top"] = pivot_walked["twap"]
    summary["proxy_pct_walked_beyond_top"] = pivot_walked[proxy_col]
    summary["winner"] = np.where(
        summary["twap_slippage_bps"] < summary["proxy_slippage_bps"],
        "TWAP",
        np.where(summary["proxy_slippage_bps"] < summary["twap_slippage_bps"], "Proxy", "Tie"),
    )
    summary["window_return_bps"] = meta_df["window_return_bps"]
    summary["avg_spread"] = meta_df["avg_spread"]
    summary["avg_relative_spread"] = meta_df["avg_relative_spread"]
    summary["avg_total_depth_5"] = meta_df["avg_total_depth_5"]
    summary["avg_abs_obi_5"] = meta_df["avg_abs_obi_5"]
    summary["mid_price_volatility"] = meta_df["mid_price_volatility"]
    summary = summary.reset_index().rename(columns={"window_id": "window_id"})

    target_dir = Path(output_dir) if output_dir is not None else REPORTS_DIR
    ensure_directory(target_dir)
    summary.to_csv(target_dir / "execution_multi_window_comparison.csv", index=False)
    return summary


def run_signal_adjusted_execution_comparison_for_window(
    market_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    side: str = "buy",
    target_qty: float = 500000.0,
    num_slices: int = 30,
    fill_method: str = "depth_walk",
    levels: int = 5,
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
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    prepared = _prepare_market_df(market_df)
    window_df = select_execution_window(prepared, start_time, end_time)
    if len(window_df) < num_slices:
        raise ValueError("Execution window does not contain enough snapshots for the requested number of slices.")

    order = ParentOrder(
        symbol=str(window_df["symbol"].iloc[0]) if "symbol" in window_df.columns else "UNKNOWN",
        side=side,
        target_qty=target_qty,
        start_time=start_time,
        end_time=end_time,
        num_slices=num_slices,
    )

    twap_schedule = generate_twap_schedule(order, window_df)
    twap_execution = simulate_execution(order, window_df, twap_schedule, fill_method=fill_method, levels=levels)
    twap_metrics = calculate_execution_metrics(order, twap_execution, window_df)
    twap_metrics["strategy"] = "twap"

    volume_col = _detect_real_volume_column(list(window_df.columns))
    proxy_schedule, proxy_strategy_name = generate_liquidity_weighted_schedule(order, window_df, volume_col=volume_col)
    proxy_execution = simulate_execution(order, window_df, proxy_schedule, fill_method=fill_method, levels=levels)
    proxy_metrics = calculate_execution_metrics(order, proxy_execution, window_df)
    proxy_metrics["strategy"] = proxy_strategy_name

    obi_schedule = generate_obi_adjusted_twap_schedule(
        order,
        window_df,
        obi_col=obi_col,
        spread_col=spread_col,
        depth_col=depth_col,
        obi_up_threshold=obi_up_threshold,
        obi_down_threshold=obi_down_threshold,
        accelerate_multiplier=accelerate_multiplier,
        slowdown_multiplier=slowdown_multiplier,
        use_regime_filter=use_regime_filter,
        spread_threshold=spread_threshold,
        depth_threshold=depth_threshold,
    )
    obi_execution = simulate_execution(order, window_df, obi_schedule, fill_method=fill_method, levels=levels)
    obi_metrics = calculate_execution_metrics(order, obi_execution, window_df)
    obi_metrics["strategy"] = "obi_adjusted_twap"

    twap_metrics.update(_window_condition_summary(window_df, start_time, end_time))
    proxy_metrics.update(_window_condition_summary(window_df, start_time, end_time))
    obi_metrics.update(_window_condition_summary(window_df, start_time, end_time))

    metrics_df = pd.DataFrame([twap_metrics, proxy_metrics, obi_metrics])
    execution_details = {
        "twap": twap_execution,
        proxy_strategy_name: proxy_execution,
        "obi_adjusted_twap": obi_execution,
        "twap_schedule": twap_schedule,
        f"{proxy_strategy_name}_schedule": proxy_schedule,
        "obi_adjusted_twap_schedule": obi_schedule,
    }
    return metrics_df, execution_details


def run_benchmark_execution_comparison_for_window(
    market_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    side: str = "buy",
    target_qty: float = 500000.0,
    num_slices: int = 30,
    fill_method: str = "depth_walk",
    levels: int = 5,
    penalty_bps: tuple[float, ...] = (0.0, 5.0, 20.0),
    random_seed: int = 42,
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
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compare simple execution schedules under the same visible-depth assumptions."""

    prepared = _prepare_market_df(market_df)
    window_df = select_execution_window(prepared, start_time, end_time)
    if len(window_df) < num_slices:
        raise ValueError("Execution window does not contain enough snapshots for the requested number of slices.")

    order = ParentOrder(
        symbol=str(window_df["symbol"].iloc[0]) if "symbol" in window_df.columns else "UNKNOWN",
        side=side,
        target_qty=target_qty,
        start_time=start_time,
        end_time=end_time,
        num_slices=num_slices,
    )

    liquidity_schedule, liquidity_strategy_name = generate_liquidity_weighted_schedule(order, window_df)
    schedules = {
        "immediate_depth_walk": generate_immediate_schedule(order, window_df),
        "twap": generate_twap_schedule(order, window_df),
        "front_loaded": generate_front_loaded_schedule(order, window_df),
        "back_loaded": generate_back_loaded_schedule(order, window_df),
        liquidity_strategy_name: liquidity_schedule,
        "random_fixed_seed": generate_random_schedule(order, window_df, seed=random_seed),
        "obi_adjusted_twap": generate_obi_adjusted_twap_schedule(
            order,
            window_df,
            obi_col=obi_col,
            spread_col=spread_col,
            depth_col=depth_col,
            obi_up_threshold=obi_up_threshold,
            obi_down_threshold=obi_down_threshold,
            accelerate_multiplier=accelerate_multiplier,
            slowdown_multiplier=slowdown_multiplier,
            use_regime_filter=use_regime_filter,
            spread_threshold=spread_threshold,
            depth_threshold=depth_threshold,
        ),
    }

    metrics: list[dict] = []
    execution_details: dict[str, pd.DataFrame] = {}
    for strategy_name, schedule in schedules.items():
        execution = simulate_execution(order, window_df, schedule, fill_method=fill_method, levels=levels)
        strategy_metrics = calculate_execution_metrics(order, execution, window_df)
        strategy_metrics["strategy"] = strategy_name
        strategy_metrics.update(_window_condition_summary(window_df, start_time, end_time))
        metrics.append(strategy_metrics)
        execution_details[strategy_name] = execution
        execution_details[f"{strategy_name}_schedule"] = schedule

    metrics_df = add_penalized_execution_costs(pd.DataFrame(metrics), penalty_bps=penalty_bps)
    return metrics_df, execution_details


def run_multi_window_benchmark_execution_analysis(
    market_df: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    side: str = "buy",
    target_qty: float = 500000.0,
    num_slices: int = 30,
    fill_method: str = "depth_walk",
    levels: int = 5,
    max_windows: int | None = 100,
    penalty_bps: tuple[float, ...] = (0.0, 5.0, 20.0),
    random_seed: int = 42,
    **schedule_kwargs,
) -> pd.DataFrame:
    prepared = _prepare_market_df(market_df)
    rows: list[dict] = []
    for window_id, (start_time, end_time) in enumerate(windows):
        if max_windows is not None and window_id >= max_windows:
            break
        metrics_df, _ = run_benchmark_execution_comparison_for_window(
            prepared,
            start_time=start_time,
            end_time=end_time,
            side=side,
            target_qty=target_qty,
            num_slices=num_slices,
            fill_method=fill_method,
            levels=levels,
            penalty_bps=penalty_bps,
            random_seed=random_seed,
            **schedule_kwargs,
        )
        for _, row in metrics_df.iterrows():
            record = row.to_dict()
            record["window_id"] = window_id
            record["start_time"] = pd.to_datetime(start_time, utc=True)
            record["end_time"] = pd.to_datetime(end_time, utc=True)
            record["target_qty"] = float(target_qty)
            rows.append(record)
    return pd.DataFrame(rows)


def run_multi_window_signal_adjusted_analysis(
    market_df: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    side: str = "buy",
    target_qty: float = 500000.0,
    num_slices: int = 30,
    fill_method: str = "depth_walk",
    levels: int = 5,
    max_windows: int | None = 100,
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
    prepared = _prepare_market_df(market_df)
    rows: list[dict] = []
    for window_id, (start_time, end_time) in enumerate(windows):
        if max_windows is not None and window_id >= max_windows:
            break
        metrics_df, _ = run_signal_adjusted_execution_comparison_for_window(
            prepared,
            start_time=start_time,
            end_time=end_time,
            side=side,
            target_qty=target_qty,
            num_slices=num_slices,
            fill_method=fill_method,
            levels=levels,
            obi_col=obi_col,
            spread_col=spread_col,
            depth_col=depth_col,
            obi_up_threshold=obi_up_threshold,
            obi_down_threshold=obi_down_threshold,
            accelerate_multiplier=accelerate_multiplier,
            slowdown_multiplier=slowdown_multiplier,
            use_regime_filter=use_regime_filter,
            spread_threshold=spread_threshold,
            depth_threshold=depth_threshold,
        )
        for _, row in metrics_df.iterrows():
            record = row.to_dict()
            record["window_id"] = window_id
            record["start_time"] = pd.to_datetime(start_time, utc=True)
            record["end_time"] = pd.to_datetime(end_time, utc=True)
            record["target_qty"] = float(target_qty)
            rows.append(record)
    return pd.DataFrame(rows)


def summarize_signal_adjusted_execution(
    results_df: pd.DataFrame,
    output_dir: str | Path | None = None,
    output_filename: str = "signal_adjusted_execution_summary.csv",
) -> pd.DataFrame:
    required = {
        "window_id",
        "strategy",
        "slippage_bps",
        "execution_cost",
        "fill_rate",
        "avg_levels_used",
        "pct_child_orders_walked_beyond_top",
    }
    missing = sorted(required - set(results_df.columns))
    if missing:
        raise ValueError(f"results_df is missing required columns: {missing}")

    agg_spec = {
        "mean_slippage_bps": ("slippage_bps", "mean"),
        "median_slippage_bps": ("slippage_bps", "median"),
        "mean_execution_cost": ("execution_cost", "mean"),
        "median_execution_cost": ("execution_cost", "median"),
        "mean_fill_rate": ("fill_rate", "mean"),
        "mean_avg_levels_used": ("avg_levels_used", "mean"),
        "mean_pct_walked_beyond_top": ("pct_child_orders_walked_beyond_top", "mean"),
    }
    if "requested_qty" in results_df.columns:
        agg_spec["mean_requested_qty"] = ("requested_qty", "mean")
    if "executed_qty" in results_df.columns:
        agg_spec["mean_executed_qty"] = ("executed_qty", "mean")
    if "unfilled_qty" in results_df.columns:
        agg_spec["mean_unfilled_qty"] = ("unfilled_qty", "mean")
    if "order_size_to_visible_depth_ratio" in results_df.columns:
        agg_spec["mean_order_size_to_visible_depth_ratio"] = ("order_size_to_visible_depth_ratio", "mean")
    for column in [col for col in results_df.columns if col.startswith("penalized_cost_")]:
        agg_spec[f"mean_{column}"] = (column, "mean")

    summary = (
        results_df.groupby("strategy", as_index=False)
        .agg(**agg_spec)
        .sort_values("strategy")
        .reset_index(drop=True)
    )

    slippage_pivot = results_df.pivot(index="window_id", columns="strategy", values="slippage_bps")
    lowest_slippage_counts: dict[str, int] = {}
    for strategy in summary["strategy"]:
        lowest_count = 0
        for _, row in slippage_pivot.iterrows():
            min_value = row.min()
            if pd.notna(row.get(strategy)) and np.isclose(row.get(strategy), min_value, equal_nan=False):
                lowest_count += 1
        lowest_slippage_counts[strategy] = lowest_count
    total_windows = int(slippage_pivot.shape[0])
    summary["lowest_slippage_count"] = summary["strategy"].map(lowest_slippage_counts).fillna(0).astype(int)
    summary["lowest_slippage_share"] = summary["lowest_slippage_count"] / total_windows if total_windows else np.nan

    target_dir = Path(output_dir) if output_dir is not None else REPORTS_DIR
    ensure_directory(target_dir)
    summary.to_csv(target_dir / output_filename, index=False)
    return summary


def summarize_signal_adjusted_pairwise_comparison(
    results_df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    required = {
        "window_id",
        "strategy",
        "slippage_bps",
        "window_return_bps",
        "avg_relative_spread",
        "avg_total_depth_5",
        "avg_abs_obi_5",
    }
    missing = sorted(required - set(results_df.columns))
    if missing:
        raise ValueError(f"results_df is missing required columns: {missing}")

    pivot = results_df.pivot(index="window_id", columns="strategy", values="slippage_bps")
    meta = (
        results_df[[
            "window_id",
            "window_return_bps",
            "avg_relative_spread",
            "avg_total_depth_5",
            "avg_abs_obi_5",
        ]]
        .drop_duplicates(subset=["window_id"])
        .set_index("window_id")
    )
    proxy_col = next((col for col in pivot.columns if col not in {"twap", "obi_adjusted_twap"}), None)
    if proxy_col is None:
        raise ValueError("Could not locate proxy strategy in results_df.")

    pairwise = pd.DataFrame(index=pivot.index)
    pairwise["twap_slippage_bps"] = pivot["twap"]
    pairwise["proxy_slippage_bps"] = pivot[proxy_col]
    pairwise["obi_adjusted_slippage_bps"] = pivot["obi_adjusted_twap"]
    pairwise["obi_minus_twap_slippage_bps"] = pairwise["obi_adjusted_slippage_bps"] - pairwise["twap_slippage_bps"]
    pairwise["obi_minus_proxy_slippage_bps"] = pairwise["obi_adjusted_slippage_bps"] - pairwise["proxy_slippage_bps"]
    pairwise["winner"] = np.where(
        (pairwise["obi_adjusted_slippage_bps"] <= pairwise["twap_slippage_bps"]) & (pairwise["obi_adjusted_slippage_bps"] <= pairwise["proxy_slippage_bps"]),
        "obi_adjusted_twap",
        np.where(
            (pairwise["twap_slippage_bps"] <= pairwise["proxy_slippage_bps"]),
            "twap",
            "proxy",
        ),
    )
    pairwise["window_return_bps"] = meta["window_return_bps"]
    pairwise["avg_relative_spread"] = meta["avg_relative_spread"]
    pairwise["avg_total_depth_5"] = meta["avg_total_depth_5"]
    pairwise["avg_abs_obi_5"] = meta["avg_abs_obi_5"]
    pairwise = pairwise.reset_index()

    target_dir = Path(output_dir) if output_dir is not None else REPORTS_DIR
    ensure_directory(target_dir)
    pairwise.to_csv(target_dir / "signal_adjusted_pairwise_comparison.csv", index=False)
    return pairwise


def summarize_signal_adjusted_schedule_diagnostics(
    schedules_df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    required = {
        "window_id",
        "adjustment_signal",
        "child_qty",
    }
    missing = sorted(required - set(schedules_df.columns))
    if missing:
        raise ValueError(f"schedules_df is missing required columns: {missing}")

    rows: list[dict] = []
    for window_id, group in schedules_df.groupby("window_id"):
        child_qty = pd.to_numeric(group["child_qty"], errors="coerce").dropna()
        counts = group["adjustment_signal"].value_counts()
        rows.append(
            {
                "window_id": window_id,
                "accelerate_count": int(counts.get("accelerate", 0)),
                "slowdown_count": int(counts.get("slowdown", 0)),
                "neutral_count": int(counts.get("neutral", 0)),
                "blocked_by_regime_filter_count": int(counts.get("blocked_by_regime_filter", 0)),
                "mean_child_qty": float(child_qty.mean()) if not child_qty.empty else np.nan,
                "std_child_qty": float(child_qty.std()) if not child_qty.empty else np.nan,
                "max_child_qty": float(child_qty.max()) if not child_qty.empty else np.nan,
                "max_child_qty_over_mean_child_qty": float(child_qty.max() / child_qty.mean()) if not child_qty.empty and float(child_qty.mean()) else np.nan,
                "sum_child_qty": float(child_qty.sum()) if not child_qty.empty else np.nan,
            }
        )

    diagnostics = pd.DataFrame(rows).sort_values("window_id").reset_index(drop=True)
    target_dir = Path(output_dir) if output_dir is not None else REPORTS_DIR
    ensure_directory(target_dir)
    diagnostics.to_csv(target_dir / "signal_adjusted_schedule_diagnostics.csv", index=False)
    return diagnostics
