"""Plotting helpers for the simplified market-making extension."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _save(output_path: str | Path | None) -> None:
    if output_path is None:
        return
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)


def plot_quotes_and_mid_price(timeline: pd.DataFrame, output_path: str | Path | None = None) -> None:
    """Plot mid-price and generated passive quotes."""

    import matplotlib.pyplot as plt

    required = {"timestamp", "mid_price", "bid_quote", "ask_quote"}
    missing = required - set(timeline.columns)
    if missing:
        raise ValueError(f"timeline is missing required columns: {sorted(missing)}")

    plot_df = timeline.head(250).copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(plot_df["timestamp"], plot_df["mid_price"], label="mid price", linewidth=1.4)
    ax.plot(plot_df["timestamp"], plot_df["bid_quote"], label="bid quote", linewidth=1.0)
    ax.plot(plot_df["timestamp"], plot_df["ask_quote"], label="ask quote", linewidth=1.0)
    ax.set_title("Quotes and Mid Price")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(output_path)
    plt.close(fig)


def plot_inventory_over_time(timeline: pd.DataFrame, output_path: str | Path | None = None) -> None:
    """Plot inventory in dataset-provided quantity units."""

    import matplotlib.pyplot as plt

    required = {"timestamp", "inventory"}
    missing = required - set(timeline.columns)
    if missing:
        raise ValueError(f"timeline is missing required columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(timeline["timestamp"], timeline["inventory"], linewidth=1.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Inventory Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Inventory (dataset-provided quantity units)")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(output_path)
    plt.close(fig)


def plot_marked_pnl_over_time(timeline: pd.DataFrame, output_path: str | Path | None = None) -> None:
    """Plot marked PnL in dataset-provided notional units."""

    import matplotlib.pyplot as plt

    required = {"timestamp", "marked_pnl"}
    missing = required - set(timeline.columns)
    if missing:
        raise ValueError(f"timeline is missing required columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(timeline["timestamp"], timeline["marked_pnl"], linewidth=1.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Marked PnL Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Marked PnL (dataset-provided notional units)")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(output_path)
    plt.close(fig)


def plot_parameter_sensitivity(sensitivity: pd.DataFrame, output_path: str | Path | None = None) -> None:
    """Plot descriptive parameter sensitivity for inventory and marked PnL."""

    import matplotlib.pyplot as plt

    required = {"sensitivity_parameter", "sensitivity_value", "total_marked_pnl", "average_absolute_inventory"}
    missing = required - set(sensitivity.columns)
    if missing:
        raise ValueError(f"sensitivity is missing required columns: {sorted(missing)}")

    parameters = list(dict.fromkeys(sensitivity["sensitivity_parameter"].astype(str)))
    fig, axes = plt.subplots(len(parameters), 2, figsize=(10, 3.2 * len(parameters)))
    if len(parameters) == 1:
        axes = [axes]

    for row_idx, parameter in enumerate(parameters):
        subset = sensitivity[sensitivity["sensitivity_parameter"] == parameter].sort_values("sensitivity_value")
        axes[row_idx][0].plot(subset["sensitivity_value"], subset["total_marked_pnl"], marker="o")
        axes[row_idx][0].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[row_idx][0].set_title(f"{parameter}: marked PnL")
        axes[row_idx][0].set_xlabel(parameter)
        axes[row_idx][0].set_ylabel("Marked PnL (dataset notional units)")

        axes[row_idx][1].plot(subset["sensitivity_value"], subset["average_absolute_inventory"], marker="o")
        axes[row_idx][1].set_title(f"{parameter}: average inventory")
        axes[row_idx][1].set_xlabel(parameter)
        axes[row_idx][1].set_ylabel("Average absolute inventory (quantity units)")

    fig.tight_layout()
    _save(output_path)
    plt.close(fig)


def plot_pnl_decomposition(summary: pd.DataFrame, output_path: str | Path | None = None) -> None:
    """Plot PnL decomposition where spread capture is a proxy."""

    import matplotlib.pyplot as plt

    required = {"strategy", "gross_spread_capture_proxy", "inventory_revaluation_pnl", "fees"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"summary is missing required columns: {sorted(missing)}")

    plot_df = summary.copy()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(plot_df["strategy"], plot_df["gross_spread_capture_proxy"], label="spread-capture proxy")
    ax.bar(
        plot_df["strategy"],
        plot_df["inventory_revaluation_pnl"],
        bottom=plot_df["gross_spread_capture_proxy"],
        label="inventory revaluation",
    )
    ax.bar(plot_df["strategy"], -plot_df["fees"], label="fees")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Marked PnL Decomposition Proxy")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("PnL components (dataset-provided notional units)")
    ax.legend()
    fig.tight_layout()
    _save(output_path)
    plt.close(fig)
