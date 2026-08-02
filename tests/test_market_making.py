from __future__ import annotations

import numpy as np
import pandas as pd

from src.market_making import (
    MarketMakingConfig,
    MarketMakingState,
    apply_fills,
    generate_quotes,
    marked_pnl,
    run_matched_comparison,
    run_market_making_simulation,
    simulate_fills,
)


def _book(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=rows, freq="s", tz="UTC"),
            "bid_p1": np.repeat(99.0, rows),
            "ask_p1": np.repeat(101.0, rows),
            "bid_q1": np.repeat(10_000.0, rows),
            "ask_q1": np.repeat(10_000.0, rows),
            "mid_price": np.repeat(100.0, rows),
            "spread": np.repeat(2.0, rows),
            "total_depth_5": np.repeat(20_000.0, rows),
        }
    )


def test_zero_inventory_skew_keeps_quote_center() -> None:
    cfg = MarketMakingConfig(half_spread_bps=10, inventory_skew_bps=5, max_inventory=100)
    quote = generate_quotes(_book().iloc[0], inventory=0.0, config=cfg)
    assert np.isclose((quote.bid_quote + quote.ask_quote) / 2, 100.0)


def test_positive_inventory_encourages_selling() -> None:
    cfg = MarketMakingConfig(half_spread_bps=10, inventory_skew_bps=10, max_inventory=100)
    flat = generate_quotes(_book().iloc[0], inventory=0.0, config=cfg)
    long = generate_quotes(_book().iloc[0], inventory=50.0, config=cfg)
    assert long.raw_bid_quote < flat.raw_bid_quote
    assert long.raw_ask_quote < flat.raw_ask_quote


def test_negative_inventory_encourages_buying() -> None:
    cfg = MarketMakingConfig(half_spread_bps=10, inventory_skew_bps=10, max_inventory=100)
    flat = generate_quotes(_book().iloc[0], inventory=0.0, config=cfg)
    short = generate_quotes(_book().iloc[0], inventory=-50.0, config=cfg)
    assert short.raw_bid_quote > flat.raw_bid_quote
    assert short.raw_ask_quote > flat.raw_ask_quote


def test_bid_quote_less_than_ask_quote() -> None:
    cfg = MarketMakingConfig(half_spread_bps=1, inventory_skew_bps=2, max_inventory=100)
    quote = generate_quotes(_book().iloc[0], inventory=25.0, config=cfg)
    assert quote.bid_quote < quote.ask_quote


def test_long_limit_disables_additional_buy() -> None:
    cfg = MarketMakingConfig(max_inventory=100)
    quote = generate_quotes(_book().iloc[0], inventory=100.0, config=cfg)
    assert not quote.bid_active
    assert quote.ask_active


def test_short_limit_disables_additional_sell() -> None:
    cfg = MarketMakingConfig(max_inventory=100)
    quote = generate_quotes(_book().iloc[0], inventory=-100.0, config=cfg)
    assert quote.bid_active
    assert not quote.ask_active


def test_executed_quantity_does_not_exceed_quoted_quantity() -> None:
    cfg = MarketMakingConfig(order_size=50, half_spread_bps=0)
    next_row = _book().iloc[1].copy()
    quote = generate_quotes(_book().iloc[0], inventory=0.0, config=cfg)
    fills = simulate_fills(next_row, quote, cfg)
    assert fills.bid_fill_qty <= cfg.order_size
    assert fills.ask_fill_qty <= cfg.order_size


def test_buy_fill_reduces_cash_and_increases_inventory() -> None:
    cfg = MarketMakingConfig(order_size=10, half_spread_bps=0)
    state = MarketMakingState()
    quote = generate_quotes(_book().iloc[0], inventory=0.0, config=cfg)
    fills = simulate_fills(_book().iloc[1], quote, cfg)
    fill_timestamp = _book().iloc[1]["timestamp"]
    apply_fills(state, quote, fills, cfg, fill_timestamp=fill_timestamp)
    assert state.inventory >= 0
    if fills.bid_filled:
        assert state.cash < 0


def test_sell_fill_increases_cash_and_reduces_inventory() -> None:
    cfg = MarketMakingConfig(order_size=10, half_spread_bps=0)
    state = MarketMakingState(inventory=10.0)
    quote = generate_quotes(_book().iloc[0], inventory=state.inventory, config=cfg)
    fills = simulate_fills(_book().iloc[1], quote, cfg)
    fill_timestamp = _book().iloc[1]["timestamp"]
    apply_fills(state, quote, fills, cfg, fill_timestamp=fill_timestamp)
    if fills.ask_filled:
        assert state.cash > 0
        assert state.inventory < 10.0


def test_total_marked_pnl_accounting_identity() -> None:
    result = run_market_making_simulation(_book(rows=8), MarketMakingConfig(half_spread_bps=0, order_size=10))
    summary = result.summary
    reconstructed = summary["cash"] + summary["ending_inventory"] * result.timeline["mid_price"].iloc[-1]
    assert np.isclose(summary["total_marked_pnl"], reconstructed)
    assert np.isclose(summary["accounting_identity_error"], 0.0)
    assert np.isclose(summary["pnl_decomposition_error"], 0.0)


def test_no_fills_marked_pnl_is_existing_inventory_mark() -> None:
    state = MarketMakingState(cash=0.0, inventory=2.0)
    assert marked_pnl(state, mark_price=110.0) == 220.0


def test_simulator_first_step_does_not_depend_on_later_snapshots() -> None:
    cfg = MarketMakingConfig(half_spread_bps=1, order_size=10, decision_interval_seconds=1)
    base = _book(rows=5)
    changed = base.copy()
    changed.loc[2:, "bid_p1"] = 10_000.0
    changed.loc[2:, "ask_p1"] = 10_001.0
    first = run_market_making_simulation(base, cfg).timeline.iloc[0]
    second = run_market_making_simulation(changed, cfg).timeline.iloc[0]
    for col in ["bid_quote", "ask_quote", "bid_fill", "ask_fill", "inventory", "cash"]:
        assert first[col] == second[col]


def test_quote_timestamp_precedes_fill_timestamp() -> None:
    result = run_market_making_simulation(_book(rows=8), MarketMakingConfig(half_spread_bps=0, order_size=10, decision_interval_seconds=1))
    first = result.timeline.iloc[0]
    assert first["quote_timestamp"] < first["fill_timestamp"]


def test_trade_timestamp_equals_fill_observation_timestamp() -> None:
    result = run_market_making_simulation(_book(rows=8), MarketMakingConfig(half_spread_bps=0, order_size=10, decision_interval_seconds=1))
    if not result.trades.empty:
        assert (result.trades["timestamp"] == result.trades["fill_timestamp"]).all()


def test_marked_pnl_uses_fill_snapshot_mid_after_fill() -> None:
    market = _book(rows=3)
    market.loc[1, "mid_price"] = 110.0
    market.loc[1, "bid_p1"] = 109.0
    market.loc[1, "ask_p1"] = 111.0
    result = run_market_making_simulation(market, MarketMakingConfig(half_spread_bps=0, order_size=10, decision_interval_seconds=1), max_steps=1)
    row = result.timeline.iloc[0]
    reconstructed = row["cash"] + row["inventory"] * market.loc[1, "mid_price"]
    assert np.isclose(row["marked_pnl"], reconstructed)
    assert row["mid_price"] == market.loc[1, "mid_price"]


def test_matched_comparison_uses_same_snapshots() -> None:
    market = _book(rows=8)
    symmetric = MarketMakingConfig(half_spread_bps=0, inventory_skew_bps=0, decision_interval_seconds=1)
    inventory = MarketMakingConfig(half_spread_bps=0, inventory_skew_bps=2, decision_interval_seconds=1)
    results, _ = run_matched_comparison(market, symmetric, inventory)
    pd.testing.assert_series_equal(
        results["symmetric"].timeline["quote_timestamp"],
        results["inventory_aware"].timeline["quote_timestamp"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        results["symmetric"].timeline["fill_timestamp"],
        results["inventory_aware"].timeline["fill_timestamp"],
        check_names=False,
    )
