"""Simplified inventory-aware market-making simulation.

This module is deliberately small and explicit. It studies passive quoting,
inventory controls, touch-based fills, and marked PnL under visible-order-book
snapshot assumptions. It is not live trading infrastructure.

Quantity fields are treated as dataset-provided quantity units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


FillModel = Literal["touch"]
MarkPriceMethod = Literal["mid"]


@dataclass(frozen=True)
class MarketMakingConfig:
    """Configuration for the public simplified simulation."""

    half_spread_bps: float = 1.0
    order_size: float = 500.0
    inventory_skew_bps: float = 0.0
    max_inventory: float = 12_500.0
    maker_fee_bps: float = 0.0
    fill_model: FillModel = "touch"
    mark_price_method: MarkPriceMethod = "mid"
    decision_interval_seconds: int = 5
    liquidation_cost_bps: float = 2.0

    def __post_init__(self) -> None:
        if self.half_spread_bps < 0:
            raise ValueError("half_spread_bps must be non-negative.")
        if self.order_size <= 0:
            raise ValueError("order_size must be positive.")
        if self.max_inventory <= 0:
            raise ValueError("max_inventory must be positive.")
        if self.maker_fee_bps < -100 or self.maker_fee_bps > 100:
            raise ValueError("maker_fee_bps is outside a reasonable display range.")
        if self.fill_model != "touch":
            raise ValueError("fill_model must be 'touch'.")
        if self.mark_price_method != "mid":
            raise ValueError("mark_price_method must be 'mid'.")
        if self.decision_interval_seconds <= 0:
            raise ValueError("decision_interval_seconds must be positive.")
        if self.liquidation_cost_bps < 0:
            raise ValueError("liquidation_cost_bps must be non-negative.")


@dataclass(frozen=True)
class Quote:
    timestamp: pd.Timestamp
    mid_price: float
    raw_bid_quote: float
    raw_ask_quote: float
    bid_quote: float
    ask_quote: float
    bid_active: bool
    ask_active: bool
    inventory_ratio: float
    skew_bps: float


@dataclass(frozen=True)
class FillResult:
    bid_filled: bool
    ask_filled: bool
    bid_fill_qty: float
    ask_fill_qty: float
    bid_fill_price: float
    ask_fill_price: float


@dataclass
class MarketMakingState:
    cash: float = 0.0
    inventory: float = 0.0
    fees_paid: float = 0.0
    trade_count: int = 0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    turnover: float = 0.0
    gross_spread_capture_proxy: float = 0.0


@dataclass(frozen=True)
class SimulationResult:
    summary: dict[str, float | int | str]
    timeline: pd.DataFrame
    trades: pd.DataFrame


def prepare_market_data(market_df: pd.DataFrame) -> pd.DataFrame:
    """Return chronologically sorted snapshots with required top-of-book fields."""

    required = {"timestamp", "bid_p1", "ask_p1"}
    missing = sorted(required - set(market_df.columns))
    if missing:
        raise ValueError(f"market_df is missing required columns: {missing}")

    df = market_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ["bid_p1", "ask_p1", "bid_q1", "ask_q1", "mid_price", "spread", "total_depth_5"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "mid_price" not in df.columns:
        df["mid_price"] = (df["bid_p1"] + df["ask_p1"]) / 2.0
    if "spread" not in df.columns:
        df["spread"] = df["ask_p1"] - df["bid_p1"]
    if "total_depth_5" not in df.columns:
        bid_qty = df["bid_q1"] if "bid_q1" in df.columns else np.nan
        ask_qty = df["ask_q1"] if "ask_q1" in df.columns else np.nan
        df["total_depth_5"] = bid_qty + ask_qty

    df = df.dropna(subset=["timestamp", "bid_p1", "ask_p1", "mid_price", "spread"])
    df = df[(df["bid_p1"] > 0) & (df["ask_p1"] > 0) & (df["ask_p1"] >= df["bid_p1"])]
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="first").reset_index(drop=True)


def select_decision_rows(market_df: pd.DataFrame, decision_interval_seconds: int) -> pd.DataFrame:
    """Sample decision rows using a time interval without looking ahead."""

    if market_df.empty:
        raise ValueError("market_df is empty.")
    if decision_interval_seconds <= 0:
        raise ValueError("decision_interval_seconds must be positive.")

    timestamps = pd.to_datetime(market_df["timestamp"], utc=True)
    selected = [0]
    next_allowed = timestamps.iloc[0] + pd.Timedelta(seconds=decision_interval_seconds)
    for idx in range(1, len(market_df)):
        if timestamps.iloc[idx] >= next_allowed:
            selected.append(idx)
            next_allowed = timestamps.iloc[idx] + pd.Timedelta(seconds=decision_interval_seconds)
    return market_df.iloc[selected].reset_index(drop=True)


def generate_quotes(row: pd.Series, inventory: float, config: MarketMakingConfig) -> Quote:
    """Generate passive bid/ask quotes from current mid-price and inventory.

    Formula:
    inventory_ratio = inventory / max_inventory
    skew_bps = inventory_skew_bps * inventory_ratio
    bid = mid * (1 - half_spread_bps / 10000 - skew_bps / 10000)
    ask = mid * (1 + half_spread_bps / 10000 - skew_bps / 10000)

    Positive inventory shifts both quotes lower: buying is discouraged and
    selling is encouraged. Negative inventory shifts both quotes higher.
    """

    mid = float(row["mid_price"])
    best_bid = float(row["bid_p1"])
    best_ask = float(row["ask_p1"])
    inventory_ratio = float(np.clip(inventory / config.max_inventory, -1.0, 1.0))
    skew_bps = config.inventory_skew_bps * inventory_ratio

    raw_bid = mid * (1.0 - config.half_spread_bps / 10000.0 - skew_bps / 10000.0)
    raw_ask = mid * (1.0 + config.half_spread_bps / 10000.0 - skew_bps / 10000.0)

    bid_quote = min(raw_bid, best_bid)
    ask_quote = max(raw_ask, best_ask)
    bid_active = inventory < config.max_inventory - 1e-12
    ask_active = inventory > -config.max_inventory + 1e-12

    if bid_quote >= ask_quote:
        bid_active = False
        ask_active = False

    return Quote(
        timestamp=pd.to_datetime(row["timestamp"], utc=True),
        mid_price=mid,
        raw_bid_quote=float(raw_bid),
        raw_ask_quote=float(raw_ask),
        bid_quote=float(bid_quote),
        ask_quote=float(ask_quote),
        bid_active=bool(bid_active),
        ask_active=bool(ask_active),
        inventory_ratio=inventory_ratio,
        skew_bps=float(skew_bps),
    )


def simulate_fills(
    next_row: pd.Series,
    quote: Quote,
    config: MarketMakingConfig,
) -> FillResult:
    """Apply deterministic one-next-snapshot touch fills.

    A bid fills if the next observed best ask is at or below the bid quote.
    An ask fills if the next observed best bid is at or above the ask quote.
    The function does not inspect later timestamps.
    """

    next_bid = float(next_row["bid_p1"])
    next_ask = float(next_row["ask_p1"])
    bid_filled = bool(quote.bid_active and next_ask <= quote.bid_quote)
    ask_filled = bool(quote.ask_active and next_bid >= quote.ask_quote)
    return FillResult(
        bid_filled=bid_filled,
        ask_filled=ask_filled,
        bid_fill_qty=config.order_size if bid_filled else 0.0,
        ask_fill_qty=config.order_size if ask_filled else 0.0,
        bid_fill_price=quote.bid_quote if bid_filled else np.nan,
        ask_fill_price=quote.ask_quote if ask_filled else np.nan,
    )


def apply_fills(
    state: MarketMakingState,
    quote: Quote,
    fills: FillResult,
    config: MarketMakingConfig,
    fill_timestamp: pd.Timestamp,
) -> list[dict[str, float | str | pd.Timestamp]]:
    """Update state for fills and return trade records."""

    trades: list[dict[str, float | str | pd.Timestamp]] = []
    fee_rate = config.maker_fee_bps / 10000.0

    if fills.bid_filled and state.inventory + fills.bid_fill_qty <= config.max_inventory + 1e-12:
        notional = fills.bid_fill_qty * fills.bid_fill_price
        fee = abs(notional) * fee_rate
        state.cash -= notional + fee
        state.inventory += fills.bid_fill_qty
        state.fees_paid += fee
        state.trade_count += 1
        state.buy_volume += fills.bid_fill_qty
        state.turnover += abs(notional)
        state.gross_spread_capture_proxy += max(quote.mid_price - fills.bid_fill_price, 0.0) * fills.bid_fill_qty
        trades.append(
            {
                "timestamp": fill_timestamp,
                "quote_timestamp": quote.timestamp,
                "fill_timestamp": fill_timestamp,
                "side": "buy",
                "price": fills.bid_fill_price,
                "quantity": fills.bid_fill_qty,
                "fee": fee,
                "inventory_after_trade": state.inventory,
            }
        )

    if fills.ask_filled and state.inventory - fills.ask_fill_qty >= -config.max_inventory - 1e-12:
        notional = fills.ask_fill_qty * fills.ask_fill_price
        fee = abs(notional) * fee_rate
        state.cash += notional - fee
        state.inventory -= fills.ask_fill_qty
        state.fees_paid += fee
        state.trade_count += 1
        state.sell_volume += fills.ask_fill_qty
        state.turnover += abs(notional)
        state.gross_spread_capture_proxy += max(fills.ask_fill_price - quote.mid_price, 0.0) * fills.ask_fill_qty
        trades.append(
            {
                "timestamp": fill_timestamp,
                "quote_timestamp": quote.timestamp,
                "fill_timestamp": fill_timestamp,
                "side": "sell",
                "price": fills.ask_fill_price,
                "quantity": fills.ask_fill_qty,
                "fee": fee,
                "inventory_after_trade": state.inventory,
            }
        )

    return trades


def marked_pnl(state: MarketMakingState, mark_price: float) -> float:
    """Marked PnL with zero initial equity."""

    return float(state.cash + state.inventory * mark_price)


def _summary_from_timeline(
    timeline: pd.DataFrame,
    trades: pd.DataFrame,
    state: MarketMakingState,
    config: MarketMakingConfig,
    strategy: str,
) -> dict[str, float | int | str]:
    final_mark = float(timeline["mid_price"].iloc[-1])
    total_marked_pnl = marked_pnl(state, final_mark)
    liquidation_cost = abs(state.inventory) * final_mark * config.liquidation_cost_bps / 10000.0
    pnl_after_liquidation = total_marked_pnl - liquidation_cost
    gross_spread = float(state.gross_spread_capture_proxy)
    inventory_revaluation = total_marked_pnl - gross_spread + state.fees_paid
    active_quote_sides = int(timeline["active_quote_sides"].sum())
    fill_count = int(timeline["bid_fill"].sum() + timeline["ask_fill"].sum())
    pnl_curve = timeline["marked_pnl"].astype(float)
    drawdown = pnl_curve - pnl_curve.cummax()
    avg_depth = float(pd.to_numeric(timeline["visible_depth"], errors="coerce").mean())

    return {
        "strategy": strategy,
        "start_time": str(timeline["timestamp"].iloc[0]),
        "end_time": str(timeline["timestamp"].iloc[-1]),
        "half_spread_bps": config.half_spread_bps,
        "order_size": config.order_size,
        "inventory_skew_bps": config.inventory_skew_bps,
        "max_inventory": config.max_inventory,
        "maker_fee_bps": config.maker_fee_bps,
        "fill_model": config.fill_model,
        "mark_price_method": config.mark_price_method,
        "total_marked_pnl": total_marked_pnl,
        "pnl_after_liquidation": pnl_after_liquidation,
        "liquidation_cost": liquidation_cost,
        "cash": float(state.cash),
        "ending_inventory": float(state.inventory),
        "ending_inventory_mark_value": float(state.inventory * final_mark),
        "gross_spread_capture_proxy": gross_spread,
        "inventory_revaluation_pnl": inventory_revaluation,
        "fees": float(state.fees_paid),
        "turnover": float(state.turnover),
        "trade_count": int(state.trade_count),
        "fill_rate": fill_count / active_quote_sides if active_quote_sides else np.nan,
        "average_absolute_inventory": float(timeline["inventory"].abs().mean()),
        "maximum_absolute_inventory": float(timeline["inventory"].abs().max()),
        "average_absolute_inventory_over_limit": float(timeline["inventory"].abs().mean() / config.max_inventory),
        "maximum_absolute_inventory_over_limit": float(timeline["inventory"].abs().max() / config.max_inventory),
        "maximum_drawdown": float(drawdown.min()),
        "pnl_volatility": float(pnl_curve.diff().dropna().std(ddof=0)),
        "average_visible_depth": avg_depth,
        "order_size_to_average_visible_depth": config.order_size / avg_depth if avg_depth else np.nan,
        "accounting_identity_error": total_marked_pnl - (state.cash + state.inventory * final_mark),
        "pnl_decomposition_error": total_marked_pnl - (gross_spread + inventory_revaluation - state.fees_paid),
    }


def run_market_making_simulation(
    market_df: pd.DataFrame,
    config: MarketMakingConfig,
    strategy_name: str = "symmetric",
    max_steps: int | None = None,
) -> SimulationResult:
    """Run the simplified simulation and return summary, timeline, and trades.

    Quotes are formed at time t and evaluated with a deterministic
    one-next-snapshot touch proxy at time t+1. Cash/inventory updates and
    marked PnL use the t+1 fill/mark observation.
    """

    prepared = prepare_market_data(market_df)
    decisions = select_decision_rows(prepared, config.decision_interval_seconds)
    if len(decisions) < 2:
        raise ValueError("Not enough decision snapshots for simulation.")
    if max_steps is not None:
        decisions = decisions.iloc[: max_steps + 1].copy()

    state = MarketMakingState()
    timeline_rows: list[dict[str, float | int | bool | str | pd.Timestamp]] = []
    trade_rows: list[dict[str, float | str | pd.Timestamp]] = []

    for idx in range(len(decisions) - 1):
        row = decisions.iloc[idx]
        next_row = decisions.iloc[idx + 1]
        quote = generate_quotes(row, state.inventory, config)
        fills = simulate_fills(next_row, quote, config)
        fill_timestamp = pd.to_datetime(next_row["timestamp"], utc=True)
        step_trades = apply_fills(state, quote, fills, config, fill_timestamp=fill_timestamp)
        trade_rows.extend(step_trades)
        quote_mid = float(row["mid_price"])
        mark = float(next_row["mid_price"])
        visible_depth = float(row["total_depth_5"]) if pd.notna(row.get("total_depth_5", np.nan)) else np.nan
        timeline_rows.append(
            {
                "timestamp": fill_timestamp,
                "quote_timestamp": quote.timestamp,
                "fill_timestamp": fill_timestamp,
                "mark_timestamp": fill_timestamp,
                "quote_mid_price": quote_mid,
                "mid_price": mark,
                "best_bid": float(row["bid_p1"]),
                "best_ask": float(row["ask_p1"]),
                "fill_best_bid": float(next_row["bid_p1"]),
                "fill_best_ask": float(next_row["ask_p1"]),
                "spread": float(row["spread"]),
                "visible_depth": visible_depth,
                "bid_quote": quote.bid_quote,
                "ask_quote": quote.ask_quote,
                "raw_bid_quote": quote.raw_bid_quote,
                "raw_ask_quote": quote.raw_ask_quote,
                "bid_quote_active": quote.bid_active,
                "ask_quote_active": quote.ask_active,
                "active_quote_sides": int(quote.bid_active) + int(quote.ask_active),
                "bid_fill": fills.bid_filled,
                "ask_fill": fills.ask_filled,
                "bid_fill_qty": fills.bid_fill_qty,
                "ask_fill_qty": fills.ask_fill_qty,
                "inventory": state.inventory,
                "cash": state.cash,
                "fees_paid": state.fees_paid,
                "marked_pnl": marked_pnl(state, mark),
                "inventory_ratio": state.inventory / config.max_inventory,
                "quote_inventory_ratio": quote.inventory_ratio,
                "skew_bps": quote.skew_bps,
                "trade_count": state.trade_count,
                "turnover": state.turnover,
                "gross_spread_capture_proxy": state.gross_spread_capture_proxy,
            }
        )

    timeline = pd.DataFrame(timeline_rows)
    trades = pd.DataFrame(
        trade_rows,
        columns=["timestamp", "side", "price", "quantity", "fee", "inventory_after_trade"],
    )
    summary = _summary_from_timeline(timeline, trades, state, config, strategy_name)
    return SimulationResult(summary=summary, timeline=timeline, trades=trades)


def run_matched_comparison(
    market_df: pd.DataFrame,
    symmetric_config: MarketMakingConfig,
    inventory_config: MarketMakingConfig,
    max_steps: int | None = None,
) -> tuple[dict[str, SimulationResult], pd.DataFrame]:
    """Run symmetric and inventory-aware simulations on the same snapshots."""

    symmetric = run_market_making_simulation(market_df, symmetric_config, "symmetric", max_steps=max_steps)
    inventory = run_market_making_simulation(market_df, inventory_config, "inventory_aware", max_steps=max_steps)
    summary = pd.DataFrame([symmetric.summary, inventory.summary])
    return {"symmetric": symmetric, "inventory_aware": inventory}, summary


def make_metric_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create a symmetric-vs-inventory-aware comparison table."""

    if set(summary_df["strategy"]) < {"symmetric", "inventory_aware"}:
        raise ValueError("summary_df must include symmetric and inventory_aware rows.")
    metrics = [
        "total_marked_pnl",
        "pnl_after_liquidation",
        "fill_rate",
        "trade_count",
        "average_absolute_inventory",
        "maximum_absolute_inventory",
        "ending_inventory",
        "gross_spread_capture_proxy",
        "inventory_revaluation_pnl",
        "fees",
        "turnover",
    ]
    indexed = summary_df.set_index("strategy")
    rows = []
    for metric in metrics:
        symmetric = indexed.loc["symmetric", metric]
        inventory = indexed.loc["inventory_aware", metric]
        rows.append(
            {
                "metric": metric,
                "symmetric": symmetric,
                "inventory_aware": inventory,
                "difference": inventory - symmetric,
            }
        )
    return pd.DataFrame(rows)


def run_sensitivity_analysis(
    market_df: pd.DataFrame,
    base_config: MarketMakingConfig,
    half_spread_values: list[float],
    inventory_skew_values: list[float],
    order_size_values: list[float],
    max_steps: int | None = None,
) -> pd.DataFrame:
    """Run a small descriptive sensitivity set without selecting an optimum."""

    rows: list[dict[str, float | int | str]] = []
    seen: set[tuple[str, float]] = set()
    scenarios = [
        *[("half_spread_bps", value) for value in half_spread_values],
        *[("inventory_skew_bps", value) for value in inventory_skew_values],
        *[("order_size", value) for value in order_size_values],
    ]
    for parameter, value in scenarios:
        key = (parameter, float(value))
        if key in seen:
            continue
        seen.add(key)
        cfg = MarketMakingConfig(
            half_spread_bps=value if parameter == "half_spread_bps" else base_config.half_spread_bps,
            order_size=value if parameter == "order_size" else base_config.order_size,
            inventory_skew_bps=value if parameter == "inventory_skew_bps" else base_config.inventory_skew_bps,
            max_inventory=base_config.max_inventory,
            maker_fee_bps=base_config.maker_fee_bps,
            fill_model=base_config.fill_model,
            mark_price_method=base_config.mark_price_method,
            decision_interval_seconds=base_config.decision_interval_seconds,
            liquidation_cost_bps=base_config.liquidation_cost_bps,
        )
        result = run_market_making_simulation(market_df, cfg, strategy_name=f"sensitivity_{parameter}", max_steps=max_steps)
        row = dict(result.summary)
        row["sensitivity_parameter"] = parameter
        row["sensitivity_value"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)
