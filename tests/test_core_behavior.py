from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_quality import check_best_bid_less_than_best_ask
from src.execution import (
    ParentOrder,
    calculate_execution_metrics,
    generate_random_schedule,
    generate_liquidity_weighted_schedule,
    simulate_execution,
    simulate_market_order_fill,
)
from src.features import add_basic_lob_features, add_depth_features
from src.labels import add_future_return_label
from src.models import time_series_train_val_test_split
from src.signal_execution import generate_obi_adjusted_twap_schedule


def _market_df(rows: int = 5) -> pd.DataFrame:
    timestamps = pd.date_range("2021-01-01", periods=rows, freq="s", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "BTC",
            "bid_p1": np.repeat(99.0, rows),
            "ask_p1": np.repeat(101.0, rows),
            "bid_q1": np.repeat(10.0, rows),
            "ask_q1": np.repeat(10.0, rows),
            "bid_p2": np.repeat(98.0, rows),
            "ask_p2": np.repeat(102.0, rows),
            "bid_q2": np.repeat(10.0, rows),
            "ask_q2": np.repeat(10.0, rows),
            "bid_p3": np.repeat(97.0, rows),
            "ask_p3": np.repeat(103.0, rows),
            "bid_q3": np.repeat(10.0, rows),
            "ask_q3": np.repeat(10.0, rows),
            "bid_p4": np.repeat(96.0, rows),
            "ask_p4": np.repeat(104.0, rows),
            "bid_q4": np.repeat(10.0, rows),
            "ask_q4": np.repeat(10.0, rows),
            "bid_p5": np.repeat(95.0, rows),
            "ask_p5": np.repeat(105.0, rows),
            "bid_q5": np.repeat(10.0, rows),
            "ask_q5": np.repeat(10.0, rows),
        }
    )
    return add_depth_features(add_basic_lob_features(df), levels=5)


def test_symmetric_order_book_obi_is_zero() -> None:
    df = _market_df()
    assert float(df["obi_1"].iloc[0]) == 0.0
    assert float(df["obi_5"].iloc[0]) == 0.0


def test_best_bid_less_than_best_ask_quality_check() -> None:
    report = check_best_bid_less_than_best_ask(_market_df())
    assert report["violations"] == 0
    assert report["bid_p1_lt_ask_p1_ratio"] == 1.0


def test_depth_walk_fill_does_not_exceed_visible_quantity() -> None:
    row = _market_df(rows=1).iloc[0]
    fill = simulate_market_order_fill(row, side="buy", quantity=75.0, levels=5)
    assert fill["filled_qty"] == 50.0
    assert fill["unfilled_qty"] == 25.0
    assert 0.0 <= fill["fill_rate"] <= 1.0


def test_zero_requested_quantity_has_no_fill_and_no_cost() -> None:
    market = _market_df()
    order = ParentOrder("BTC", "buy", 0.0, market["timestamp"].iloc[0], market["timestamp"].iloc[-1], 3)
    schedule = pd.DataFrame({"timestamp": [market["timestamp"].iloc[0]], "child_qty": [0.0], "strategy": ["zero"]})
    execution = simulate_execution(order, market, schedule, fill_method="depth_walk")
    metrics = calculate_execution_metrics(order, execution, market)
    assert metrics["requested_qty"] == 0.0
    assert metrics["executed_qty"] == 0.0
    assert metrics["execution_cost"] == 0.0


def test_larger_order_size_does_not_lower_static_book_slippage() -> None:
    row = _market_df(rows=1).iloc[0]
    small = simulate_market_order_fill(row, side="buy", quantity=5.0, levels=5)
    large = simulate_market_order_fill(row, side="buy", quantity=25.0, levels=5)
    arrival_price = float(row["mid_price"])
    small_slippage = small["avg_fill_price"] - arrival_price
    large_slippage = large["avg_fill_price"] - arrival_price
    assert large_slippage >= small_slippage


def test_chronological_split_has_no_overlap() -> None:
    df = _market_df(rows=100)
    train, val, test = time_series_train_val_test_split(df)
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()


def test_feature_timestamp_precedes_future_target_timestamp() -> None:
    df = _market_df(rows=10)
    labelled = add_future_return_label(df, horizon_seconds=3, threshold=0.0001)
    assert labelled["timestamp"].iloc[0] < df["timestamp"].iloc[3]


def test_liquidity_schedule_prefix_does_not_depend_on_future_snapshot() -> None:
    market = _market_df(rows=4)
    order = ParentOrder("BTC", "buy", 100.0, market["timestamp"].iloc[0], market["timestamp"].iloc[-1], 4)
    base_schedule, _ = generate_liquidity_weighted_schedule(order, market)
    changed_future = market.copy()
    changed_future.loc[3, "ask_depth_5"] = 1000000.0
    changed_schedule, _ = generate_liquidity_weighted_schedule(order, changed_future)
    pd.testing.assert_series_equal(base_schedule["child_qty"].iloc[:3], changed_schedule["child_qty"].iloc[:3], check_names=False)


def test_obi_adjusted_schedule_prefix_does_not_depend_on_future_snapshot() -> None:
    market = _market_df(rows=4)
    market["obi_5"] = [-0.3, 0.3, 0.3, 0.3]
    order = ParentOrder("BTC", "buy", 100.0, market["timestamp"].iloc[0], market["timestamp"].iloc[-1], 4)
    base_schedule = generate_obi_adjusted_twap_schedule(order, market)
    changed_future = market.copy()
    changed_future.loc[3, "relative_spread"] = 999.0
    changed_future.loc[3, "total_depth_5"] = 0.0
    changed_schedule = generate_obi_adjusted_twap_schedule(order, changed_future)
    pd.testing.assert_series_equal(base_schedule["child_qty"].iloc[:3], changed_schedule["child_qty"].iloc[:3], check_names=False)


def test_fill_rate_is_within_unit_interval() -> None:
    row = _market_df(rows=1).iloc[0]
    fill = simulate_market_order_fill(row, side="sell", quantity=25.0, levels=5)
    assert 0.0 <= fill["fill_rate"] <= 1.0


def test_random_schedule_is_reproducible_with_fixed_seed() -> None:
    market = _market_df(rows=5)
    order = ParentOrder("BTC", "buy", 100.0, market["timestamp"].iloc[0], market["timestamp"].iloc[-1], 5)
    first = generate_random_schedule(order, market, seed=7)
    second = generate_random_schedule(order, market, seed=7)
    pd.testing.assert_frame_equal(first, second)
