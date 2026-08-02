"""Matplotlib plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _plt():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("matplotlib is required for plotting but is not available in this environment.") from exc
    return plt


def _save_and_show(output_path: str | Path | None) -> None:
    plt = _plt()
    backend = str(plt.get_backend()).lower()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
    if output_path is None and "agg" not in backend:
        plt.show()
    plt.close()


def plot_obi_distribution(df: pd.DataFrame, column: str = "obi_5", output_path: str | Path | None = None) -> None:
    plt = _plt()
    data = df[column].dropna()
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=50, color="#1f77b4", edgecolor="white", alpha=0.85)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    _save_and_show(output_path)


def plot_label_distribution(df: pd.DataFrame, output_path: str | Path | None = None) -> None:
    plt = _plt()
    order = ["Down", "Flat", "Up"]
    counts = df["label"].value_counts().reindex(order).fillna(0)
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values, color=["#d62728", "#7f7f7f", "#2ca02c"])
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    _save_and_show(output_path)


def plot_obi_decile_future_return(
    df: pd.DataFrame,
    obi_col: str = "obi_5",
    ret_col: str = "future_return_30s",
    output_path: str | Path | None = None,
) -> None:
    plt = _plt()
    plot_df = df[[obi_col, ret_col]].dropna().copy()
    if plot_df.empty:
        raise ValueError("No data available for OBI decile plot.")
    plot_df["obi_decile"] = pd.qcut(plot_df[obi_col], q=10, duplicates="drop")
    summary = plot_df.groupby("obi_decile", observed=True)[ret_col].mean()
    labels = [str(interval) for interval in summary.index]
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(summary)), summary.values, marker="o", color="#1f77b4")
    plt.xticks(range(len(summary)), labels, rotation=45, ha="right")
    plt.title(f"{obi_col} Decile vs Average {ret_col}")
    plt.xlabel(f"{obi_col} decile")
    plt.ylabel(f"Average {ret_col}")
    plt.tight_layout()
    _save_and_show(output_path)


def plot_confusion_matrix(
    cm,
    labels,
    output_path=None,
    title=None,
):
    plt = _plt()
    cm_array = np.asarray(cm)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_array, interpolation="nearest", cmap="Blues")
    plt.title(title or "Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    thresh = cm_array.max() / 2 if cm_array.size else 0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            plt.text(
                j,
                i,
                format(cm_array[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm_array[i, j] > thresh else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    _save_and_show(output_path)


def _get_xgb_core_model(model):
    core = getattr(model, "model", model)
    if hasattr(core, "get_booster"):
        return core
    if hasattr(core, "model") and hasattr(core.model, "get_booster"):
        return core.model
    return core


def plot_xgboost_feature_importance(
    model,
    feature_cols,
    output_path=None,
):
    plt = _plt()
    xgb_model = _get_xgb_core_model(model)
    booster = xgb_model.get_booster()
    scores = booster.get_score(importance_type="weight")
    importances = np.array([scores.get(f"f{i}", 0.0) for i in range(len(feature_cols))], dtype=float)
    order = np.argsort(importances)
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(feature_cols)[order], importances[order], color="#1f77b4")
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    _save_and_show(output_path)


def plot_model_metric_comparison(metrics_df, output_path=None):
    plt = _plt()
    metric_cols = [col for col in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"] if col in metrics_df.columns]
    if not metric_cols:
        raise ValueError("metrics_df must contain at least one metric column.")
    plot_df = metrics_df.copy()
    x = np.arange(len(plot_df))
    width = 0.8 / len(metric_cols)
    plt.figure(figsize=(10, 5))
    for idx, metric in enumerate(metric_cols):
        plt.bar(x + idx * width, plot_df[metric].values, width=width, label=metric)
    plt.xticks(
        x + width * (len(metric_cols) - 1) / 2,
        plot_df["model"].astype(str) + " / " + plot_df["sample"].astype(str),
        rotation=45,
        ha="right",
    )
    plt.ylabel("Score")
    plt.title("Model Metric Comparison")
    plt.legend()
    plt.tight_layout()
    _save_and_show(output_path)


def _normalize_execution_results(execution_results: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    normalized: dict[str, pd.DataFrame] = {}
    for name, df in execution_results.items():
        if df is None or df.empty:
            continue
        result = df.copy()
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
        result = result.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        normalized[name] = result
    return normalized


def plot_cumulative_execution(
    execution_results: dict[str, pd.DataFrame],
    output_path=None,
):
    plt = _plt()
    execution_results = _normalize_execution_results(execution_results)
    if not execution_results:
        raise ValueError("execution_results is empty.")
    plt.figure(figsize=(10, 5))
    for strategy, df in execution_results.items():
        if "cum_qty" not in df.columns:
            raise ValueError(f"{strategy} must contain cum_qty.")
        plt.plot(df["timestamp"], df["cum_qty"], marker="o", label=strategy)
    plt.title("Cumulative Executed Quantity")
    plt.xlabel("Timestamp")
    plt.ylabel("Cumulative Quantity")
    plt.legend()
    plt.tight_layout()
    _save_and_show(output_path)


def plot_execution_price_path(
    market_df: pd.DataFrame,
    execution_results: dict[str, pd.DataFrame],
    output_path=None,
):
    plt = _plt()
    if "timestamp" not in market_df.columns or "mid_price" not in market_df.columns:
        raise ValueError("market_df must contain timestamp and mid_price.")
    market = market_df.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True, errors="coerce")
    market = market.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    execution_results = _normalize_execution_results(execution_results)
    if market.empty:
        raise ValueError("market_df is empty after timestamp parsing.")
    plt.figure(figsize=(10, 5))
    plt.plot(market["timestamp"], market["mid_price"], color="#1f77b4", linewidth=1.5, label="mid_price")
    colors = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    for idx, (strategy, df) in enumerate(execution_results.items()):
        if "execution_price" not in df.columns:
            raise ValueError(f"{strategy} must contain execution_price.")
        plt.scatter(
            df["timestamp"],
            df["execution_price"],
            s=28,
            alpha=0.85,
            color=colors[idx % len(colors)],
            label=f"{strategy} executions",
        )
    plt.title("Mid-Price Path with Execution Points")
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    _save_and_show(output_path)


def plot_slippage_comparison(
    metrics_df: pd.DataFrame,
    output_path=None,
):
    plt = _plt()
    if "strategy" not in metrics_df.columns or "slippage_bps" not in metrics_df.columns:
        raise ValueError("metrics_df must contain strategy and slippage_bps columns.")
    plot_df = metrics_df.copy()
    plot_df = plot_df.dropna(subset=["strategy", "slippage_bps"])
    if plot_df.empty:
        raise ValueError("metrics_df is empty after dropping missing values.")
    plt.figure(figsize=(8, 4))
    plt.bar(plot_df["strategy"].astype(str), plot_df["slippage_bps"], color="#1f77b4")
    plt.title("Slippage Comparison by Strategy")
    plt.xlabel("Strategy")
    plt.ylabel("Slippage (bps)")
    plt.tight_layout()
    _save_and_show(output_path)


def plot_child_order_sizes(
    execution_results: dict[str, pd.DataFrame],
    output_path=None,
):
    plt = _plt()
    execution_results = _normalize_execution_results(execution_results)
    if not execution_results:
        raise ValueError("execution_results is empty.")
    plt.figure(figsize=(10, 5))
    for strategy, df in execution_results.items():
        if "child_qty" not in df.columns:
            raise ValueError(f"{strategy} must contain child_qty.")
        plt.plot(df["timestamp"], df["child_qty"], marker="o", label=strategy)
    plt.title("Child Order Sizes Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Child Quantity")
    plt.legend()
    plt.tight_layout()
    _save_and_show(output_path)


def plot_multi_window_slippage_distribution(comparison_df: pd.DataFrame, output_path=None):
    plt = _plt()
    if "proxy_minus_twap_slippage_bps" not in comparison_df.columns:
        raise ValueError("comparison_df must contain proxy_minus_twap_slippage_bps.")
    data = comparison_df["proxy_minus_twap_slippage_bps"].dropna()
    if data.empty:
        raise ValueError("No data available for slippage distribution plot.")
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=30, color="#1f77b4", edgecolor="white", alpha=0.85)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Proxy Minus TWAP Slippage Distribution")
    plt.xlabel("proxy_minus_twap_slippage_bps")
    plt.ylabel("Count")
    plt.tight_layout()
    _save_and_show(output_path)


def plot_strategy_lowest_slippage_count(comparison_df: pd.DataFrame, output_path=None):
    plt = _plt()
    if "winner" not in comparison_df.columns:
        raise ValueError("comparison_df must contain winner.")
    counts = comparison_df["winner"].value_counts().reindex(["TWAP", "Proxy", "Tie"]).fillna(0)
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values, color=["#d62728", "#2ca02c", "#7f7f7f"])
    plt.title("Lowest Slippage Count")
    plt.xlabel("Winner")
    plt.ylabel("Window Count")
    plt.tight_layout()
    _save_and_show(output_path)


def plot_slippage_difference_vs_window_return(comparison_df: pd.DataFrame, output_path=None):
    plt = _plt()
    required = {"window_return_bps", "proxy_minus_twap_slippage_bps"}
    if not required.issubset(comparison_df.columns):
        raise ValueError("comparison_df must contain window_return_bps and proxy_minus_twap_slippage_bps.")
    plot_df = comparison_df[list(required)].dropna()
    if plot_df.empty:
        raise ValueError("No data available for window return scatter plot.")
    plt.figure(figsize=(8, 4))
    plt.scatter(plot_df["window_return_bps"], plot_df["proxy_minus_twap_slippage_bps"], alpha=0.75, color="#1f77b4")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Slippage Difference vs Window Return")
    plt.xlabel("window_return_bps")
    plt.ylabel("proxy_minus_twap_slippage_bps")
    plt.tight_layout()
    _save_and_show(output_path)


def plot_slippage_difference_vs_depth(comparison_df: pd.DataFrame, output_path=None):
    plt = _plt()
    required = {"avg_total_depth_5", "proxy_minus_twap_slippage_bps"}
    if not required.issubset(comparison_df.columns):
        raise ValueError("comparison_df must contain avg_total_depth_5 and proxy_minus_twap_slippage_bps.")
    plot_df = comparison_df[list(required)].dropna()
    if plot_df.empty:
        raise ValueError("No data available for depth scatter plot.")
    plt.figure(figsize=(8, 4))
    plt.scatter(plot_df["avg_total_depth_5"], plot_df["proxy_minus_twap_slippage_bps"], alpha=0.75, color="#2ca02c")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Slippage Difference vs Depth")
    plt.xlabel("avg_total_depth_5")
    plt.ylabel("proxy_minus_twap_slippage_bps")
    plt.tight_layout()
    _save_and_show(output_path)


def _grouped_strategy_plot(
    comparison_df: pd.DataFrame,
    metric_cols: list[str],
    title: str,
    ylabel: str,
    output_path=None,
):
    plt = _plt()
    required = {"fill_method", *metric_cols}
    if not required.issubset(comparison_df.columns):
        raise ValueError(f"comparison_df must contain {sorted(required)}.")
    grouped = comparison_df.groupby("fill_method")[metric_cols].mean(numeric_only=True)
    if grouped.empty:
        raise ValueError("No data available for grouped strategy plot.")
    x = np.arange(len(grouped.index))
    width = 0.8 / len(metric_cols)
    plt.figure(figsize=(9, 4))
    for idx, metric in enumerate(metric_cols):
        plt.bar(x + idx * width, grouped[metric].values, width=width, label=metric)
    plt.xticks(x + width * (len(metric_cols) - 1) / 2, grouped.index.astype(str))
    plt.title(title)
    plt.xlabel("fill_method")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    _save_and_show(output_path)


def plot_depth_walk_slippage_comparison(comparison_df: pd.DataFrame, output_path=None):
    _grouped_strategy_plot(
        comparison_df,
        ["twap_slippage_bps", "proxy_slippage_bps"],
        title="Average Slippage by Fill Method",
        ylabel="Slippage (bps)",
        output_path=output_path,
    )


def plot_pct_walked_beyond_top(comparison_df: pd.DataFrame, output_path=None):
    _grouped_strategy_plot(
        comparison_df,
        ["twap_pct_walked_beyond_top", "proxy_pct_walked_beyond_top"],
        title="Pct Child Orders Walking Beyond Top of Book",
        ylabel="Percentage",
        output_path=output_path,
    )


def plot_fill_rate_by_strategy(comparison_df: pd.DataFrame, output_path=None):
    _grouped_strategy_plot(
        comparison_df,
        ["twap_fill_rate", "proxy_fill_rate"],
        title="Average Fill Rate by Fill Method",
        ylabel="Fill Rate",
        output_path=output_path,
    )


def _line_by_target_plot(
    summary_df: pd.DataFrame,
    x_col: str,
    metric_cols: list[str],
    title: str,
    ylabel: str,
    output_path=None,
    y_limits: tuple[float | None, float | None] | None = None,
):
    plt = _plt()
    required = {x_col, *metric_cols}
    if not required.issubset(summary_df.columns):
        raise ValueError(f"summary_df must contain {sorted(required)}.")
    plot_df = summary_df.sort_values(x_col).dropna(subset=[x_col])
    if plot_df.empty:
        raise ValueError("No data available for the requested plot.")
    plt.figure(figsize=(9, 4))
    for metric in metric_cols:
        plt.plot(plot_df[x_col], plot_df[metric], marker="o", linewidth=1.8, label=metric)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    if y_limits is not None:
        plt.ylim(*y_limits)
    plt.legend()
    plt.tight_layout()
    _save_and_show(output_path)


def plot_depth_stress_slippage_by_size(summary_df: pd.DataFrame, output_path=None):
    _line_by_target_plot(
        summary_df,
        x_col="target_qty",
        metric_cols=["twap_mean_slippage_bps", "proxy_mean_slippage_bps"],
        title="Average Slippage by Parent Order Size",
        ylabel="Slippage (bps)",
        output_path=output_path,
    )


def plot_depth_stress_walked_beyond_top_by_size(summary_df: pd.DataFrame, output_path=None):
    _line_by_target_plot(
        summary_df,
        x_col="target_qty",
        metric_cols=["twap_mean_pct_walked_beyond_top", "proxy_mean_pct_walked_beyond_top"],
        title="Pct Child Orders Walking Beyond Top of Book",
        ylabel="Percentage",
        output_path=output_path,
    )


def plot_depth_stress_fill_rate_by_size(summary_df: pd.DataFrame, output_path=None):
    _line_by_target_plot(
        summary_df,
        x_col="target_qty",
        metric_cols=["twap_mean_fill_rate", "proxy_mean_fill_rate"],
        title="Average Fill Rate by Parent Order Size",
        ylabel="Fill Rate",
        output_path=output_path,
        y_limits=(0.0, 1.05),
    )


def plot_depth_stress_avg_levels_used_by_size(summary_df: pd.DataFrame, output_path=None):
    _line_by_target_plot(
        summary_df,
        x_col="target_qty",
        metric_cols=["twap_mean_avg_levels_used", "proxy_mean_avg_levels_used"],
        title="Average Visible Levels Used by Parent Order Size",
        ylabel="Average Levels Used",
        output_path=output_path,
    )


def plot_depth_stress_proxy_minus_twap_by_size(summary_df: pd.DataFrame, output_path=None):
    plt = _plt()
    required = {"target_qty", "mean_proxy_minus_twap_slippage_bps"}
    if not required.issubset(summary_df.columns):
        raise ValueError("summary_df must contain target_qty and mean_proxy_minus_twap_slippage_bps.")
    plot_df = summary_df.sort_values("target_qty").dropna(subset=["target_qty", "mean_proxy_minus_twap_slippage_bps"])
    if plot_df.empty:
        raise ValueError("No data available for proxy-minus-TWAP plot.")
    plt.figure(figsize=(9, 4))
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.plot(
        plot_df["target_qty"],
        plot_df["mean_proxy_minus_twap_slippage_bps"],
        marker="o",
        linewidth=1.8,
        color="#d62728",
    )
    plt.title("Proxy Minus TWAP Slippage by Parent Order Size")
    plt.xlabel("target_qty")
    plt.ylabel("proxy_minus_twap_slippage_bps")
    plt.tight_layout()
    _save_and_show(output_path)


def plot_depth_stress_child_order_concentration(concentration_df: pd.DataFrame, output_path=None):
    plt = _plt()
    required = {"target_qty", "strategy", "mean_child_qty", "max_child_qty_over_mean_child_qty"}
    if not required.issubset(concentration_df.columns):
        raise ValueError(f"concentration_df must contain {sorted(required)}.")
    plot_df = concentration_df.sort_values(["target_qty", "strategy"]).dropna(subset=["target_qty", "mean_child_qty"])
    if plot_df.empty:
        raise ValueError("No data available for child order concentration plot.")
    plt.figure(figsize=(9, 4))
    for strategy, group in plot_df.groupby("strategy"):
        plt.plot(group["target_qty"], group["max_child_qty_over_mean_child_qty"], marker="o", linewidth=1.8, label=strategy)
    plt.title("Child Order Concentration by Parent Order Size")
    plt.xlabel("target_qty")
    plt.ylabel("max_child_qty / mean_child_qty")
    plt.legend()
    plt.tight_layout()
    _save_and_show(output_path)


def plot_signal_adjusted_strategy_slippage(summary_df: pd.DataFrame, output_path=None):
    plt = _plt()
    required = {"strategy", "mean_slippage_bps"}
    if not required.issubset(summary_df.columns):
        raise ValueError(f"summary_df must contain {sorted(required)}.")
    plot_df = summary_df.dropna(subset=["strategy", "mean_slippage_bps"]).copy()
    if plot_df.empty:
        raise ValueError("No data available for strategy slippage plot.")
    plt.figure(figsize=(8, 4))
    plt.bar(plot_df["strategy"].astype(str), plot_df["mean_slippage_bps"], color="#1f77b4")
    plt.title("Mean Slippage by Strategy")
    plt.xlabel("Strategy")
    plt.ylabel("Mean slippage (bps)")
    plt.tight_layout()
    _save_and_show(output_path)


def plot_signal_adjusted_lowest_slippage_share(summary_df: pd.DataFrame, output_path=None):
    plt = _plt()
    required = {"strategy", "lowest_slippage_share"}
    if not required.issubset(summary_df.columns):
        raise ValueError(f"summary_df must contain {sorted(required)}.")
    plot_df = summary_df.dropna(subset=["strategy", "lowest_slippage_share"]).copy()
    if plot_df.empty:
        raise ValueError("No data available for lowest-slippage share plot.")
    plt.figure(figsize=(8, 4))
    plt.bar(plot_df["strategy"].astype(str), plot_df["lowest_slippage_share"], color="#2ca02c")
    plt.title("Share of Windows with Lowest Slippage")
    plt.xlabel("Strategy")
    plt.ylabel("Share of windows")
    plt.ylim(0, 1)
    plt.tight_layout()
    _save_and_show(output_path)


def plot_signal_adjusted_pairwise_vs_twap(pairwise_df: pd.DataFrame, output_path=None):
    plt = _plt()
    required = {"window_id", "twap_slippage_bps", "proxy_slippage_bps", "obi_adjusted_slippage_bps"}
    if not required.issubset(pairwise_df.columns):
        raise ValueError(f"pairwise_df must contain {sorted(required)}.")
    plot_df = pairwise_df.sort_values("window_id").dropna(subset=["window_id"])
    if plot_df.empty:
        raise ValueError("No data available for pairwise plot.")
    plt.figure(figsize=(10, 4))
    plt.plot(plot_df["window_id"], plot_df["twap_slippage_bps"], marker="o", linewidth=1.3, label="twap")
    plt.plot(plot_df["window_id"], plot_df["proxy_slippage_bps"], marker="o", linewidth=1.3, label="proxy")
    plt.plot(plot_df["window_id"], plot_df["obi_adjusted_slippage_bps"], marker="o", linewidth=1.3, label="obi_adjusted_twap")
    plt.title("Pairwise Slippage by Window")
    plt.xlabel("window_id")
    plt.ylabel("Slippage (bps)")
    plt.legend()
    plt.tight_layout()
    _save_and_show(output_path)


def plot_signal_adjusted_schedule_concentration(schedule_df: pd.DataFrame, output_path=None):
    plt = _plt()
    required = {"window_id", "mean_child_qty", "max_child_qty_over_mean_child_qty"}
    if not required.issubset(schedule_df.columns):
        raise ValueError(f"schedule_df must contain {sorted(required)}.")
    plot_df = schedule_df.sort_values("window_id").dropna(subset=["window_id"])
    if plot_df.empty:
        raise ValueError("No data available for schedule concentration plot.")
    plt.figure(figsize=(10, 4))
    plt.plot(plot_df["window_id"], plot_df["max_child_qty_over_mean_child_qty"], marker="o", linewidth=1.5, color="#d62728")
    plt.title("Signal-Adjusted Schedule Concentration")
    plt.xlabel("window_id")
    plt.ylabel("max_child_qty / mean_child_qty")
    plt.tight_layout()
    _save_and_show(output_path)


def plot_signal_adjusted_slippage_vs_window_return(pairwise_df: pd.DataFrame, output_path=None):
    plt = _plt()
    required = {"window_return_bps", "obi_minus_twap_slippage_bps"}
    if not required.issubset(pairwise_df.columns):
        raise ValueError(f"pairwise_df must contain {sorted(required)}.")
    plot_df = pairwise_df.dropna(subset=["window_return_bps", "obi_minus_twap_slippage_bps"])
    if plot_df.empty:
        raise ValueError("No data available for slippage vs window return plot.")
    plt.figure(figsize=(8, 4))
    plt.scatter(plot_df["window_return_bps"], plot_df["obi_minus_twap_slippage_bps"], alpha=0.75, color="#1f77b4")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("OBI Adjusted Minus TWAP Slippage vs Window Return")
    plt.xlabel("window_return_bps")
    plt.ylabel("obi_minus_twap_slippage_bps")
    plt.tight_layout()
    _save_and_show(output_path)


def plot_signal_robustness_obi_minus_twap_distribution(summary_df: pd.DataFrame, output_path=None):
    plt = _plt()
    required = {"obi_minus_twap_mean_slippage_bps"}
    if not required.issubset(summary_df.columns):
        raise ValueError(f"summary_df must contain {sorted(required)}.")
    data = summary_df["obi_minus_twap_mean_slippage_bps"].dropna()
    if data.empty:
        raise ValueError("No data available for robustness distribution plot.")
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=20, color="#1f77b4", edgecolor="white", alpha=0.85)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Distribution of OBI Minus TWAP Mean Slippage")
    plt.xlabel("obi_minus_twap_mean_slippage_bps")
    plt.ylabel("Count")
    plt.tight_layout()
    _save_and_show(output_path)


def _plot_grouped_metric(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path=None,
):
    plt = _plt()
    required = {x_col, y_col}
    if not required.issubset(df.columns):
        raise ValueError(f"df must contain {sorted(required)}.")
    plot_df = df[[x_col, y_col]].dropna().sort_values(x_col)
    if plot_df.empty:
        raise ValueError("No data available for grouped metric plot.")
    plt.figure(figsize=(8, 4))
    plt.plot(plot_df[x_col], plot_df[y_col], marker="o", linewidth=1.8, color="#1f77b4")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.tight_layout()
    _save_and_show(output_path)


def plot_signal_robustness_by_target_qty(df: pd.DataFrame, output_path=None):
    _plot_grouped_metric(
        df,
        x_col="target_qty",
        y_col="mean_obi_minus_twap_mean_slippage_bps",
        title="Robustness by Target Quantity",
        ylabel="Mean obi_minus_twap_mean_slippage_bps",
        output_path=output_path,
    )


def plot_signal_robustness_by_threshold(df: pd.DataFrame, output_path=None):
    _plot_grouped_metric(
        df,
        x_col="obi_threshold",
        y_col="mean_obi_minus_twap_mean_slippage_bps",
        title="Robustness by OBI Threshold",
        ylabel="Mean obi_minus_twap_mean_slippage_bps",
        output_path=output_path,
    )


def plot_signal_robustness_by_multiplier(df: pd.DataFrame, output_path=None):
    plt = _plt()
    required = {"accelerate_multiplier", "slowdown_multiplier", "mean_obi_minus_twap_mean_slippage_bps"}
    if not required.issubset(df.columns):
        raise ValueError(f"df must contain {sorted(required)}.")
    plot_df = df.dropna(subset=list(required)).copy()
    if plot_df.empty:
        raise ValueError("No data available for multiplier plot.")
    labels = plot_df["accelerate_multiplier"].astype(str) + "/" + plot_df["slowdown_multiplier"].astype(str)
    plt.figure(figsize=(9, 4))
    plt.bar(labels, plot_df["mean_obi_minus_twap_mean_slippage_bps"], color="#2ca02c")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Robustness by Multiplier Pair")
    plt.xlabel("accelerate/slowdown")
    plt.ylabel("Mean obi_minus_twap_mean_slippage_bps")
    plt.tight_layout()
    _save_and_show(output_path)


def plot_signal_robustness_by_filter(df: pd.DataFrame, output_path=None):
    plt = _plt()
    required = {"use_regime_filter", "mean_obi_minus_twap_mean_slippage_bps"}
    if not required.issubset(df.columns):
        raise ValueError(f"df must contain {sorted(required)}.")
    plot_df = df.dropna(subset=list(required)).copy()
    if plot_df.empty:
        raise ValueError("No data available for filter plot.")
    plt.figure(figsize=(6, 4))
    plt.bar(plot_df["use_regime_filter"].astype(str), plot_df["mean_obi_minus_twap_mean_slippage_bps"], color="#ff7f0e")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Robustness by Regime Filter")
    plt.xlabel("use_regime_filter")
    plt.ylabel("Mean obi_minus_twap_mean_slippage_bps")
    plt.tight_layout()
    _save_and_show(output_path)


def plot_signal_robustness_fill_rate_vs_slippage(summary_df: pd.DataFrame, output_path=None):
    plt = _plt()
    required = {"obi_mean_fill_rate", "obi_minus_twap_mean_slippage_bps"}
    if not required.issubset(summary_df.columns):
        raise ValueError(f"summary_df must contain {sorted(required)}.")
    plot_df = summary_df.dropna(subset=list(required)).copy()
    if plot_df.empty:
        raise ValueError("No data available for fill rate vs slippage plot.")
    plt.figure(figsize=(8, 4))
    plt.scatter(plot_df["obi_mean_fill_rate"], plot_df["obi_minus_twap_mean_slippage_bps"], alpha=0.75, color="#d62728")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("obi_mean_fill_rate")
    plt.ylabel("obi_minus_twap_mean_slippage_bps")
    plt.title("Fill Rate vs OBI Minus TWAP Slippage")
    plt.tight_layout()
    _save_and_show(output_path)
