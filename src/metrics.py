"""Evaluation helpers for model comparison."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def rule_based_obi_predict(
    df: pd.DataFrame,
    obi_col: str = "obi_5",
    up_threshold: float = 0.2,
    down_threshold: float = -0.2,
) -> pd.Series:
    """Predict Up/Down/Flat using a simple OBI threshold rule."""

    if obi_col not in df.columns:
        raise ValueError(f"Missing OBI column: {obi_col}")

    preds = np.where(
        df[obi_col] > up_threshold,
        "Up",
        np.where(df[obi_col] < down_threshold, "Down", "Flat"),
    )
    return pd.Series(preds, index=df.index, name="prediction")


def majority_class_predict(y_train, y_test) -> pd.Series:
    """Predict the most frequent class observed in y_train for every test sample."""

    train_series = pd.Series(y_train).dropna()
    if train_series.empty:
        raise ValueError("y_train is empty; cannot compute majority class baseline.")
    majority_class = Counter(train_series.astype(str)).most_common(1)[0][0]
    return pd.Series([majority_class] * len(y_test), index=getattr(y_test, "index", None), name="prediction")


def evaluate_predictions(y_true, y_pred, labels=None) -> dict:
    """Return standard classification metrics and confusion matrix."""

    try:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

        if labels is None:
            labels = ["Down", "Flat", "Up"]

        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report_text = classification_report(y_true, y_pred, labels=labels, zero_division=0)
        report_dict = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    except Exception:
        labels = list(labels or ["Down", "Flat", "Up"])
        y_true_arr = pd.Series(y_true).astype(str).to_numpy()
        y_pred_arr = pd.Series(y_pred).astype(str).to_numpy()
        accuracy = float((y_true_arr == y_pred_arr).mean())
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        for truth, pred in zip(y_true_arr, y_pred_arr):
            if truth in label_to_idx and pred in label_to_idx:
                cm[label_to_idx[truth], label_to_idx[pred]] += 1
        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)
        precision = np.divide(np.diag(cm), col_sums, out=np.zeros(len(labels), dtype=float), where=col_sums != 0)
        recall = np.divide(np.diag(cm), row_sums, out=np.zeros(len(labels), dtype=float), where=row_sums != 0)
        f1_per_class = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(len(labels), dtype=float),
            where=(precision + recall) != 0,
        )
        balanced_accuracy = float(np.nanmean(recall))
        macro_f1 = float(np.nanmean(f1_per_class))
        weighted_f1 = float(np.average(f1_per_class, weights=row_sums)) if row_sums.sum() else 0.0
        report_dict = {
            label: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1-score": float(f1_per_class[idx]),
                "support": int(row_sums[idx]),
            }
            for idx, label in enumerate(labels)
        }
        report_dict["accuracy"] = accuracy
        report_dict["macro avg"] = {
            "precision": float(np.nanmean(precision)),
            "recall": float(np.nanmean(recall)),
            "f1-score": macro_f1,
            "support": int(row_sums.sum()),
        }
        report_dict["weighted avg"] = {
            "precision": float(np.average(precision, weights=row_sums)) if row_sums.sum() else 0.0,
            "recall": float(np.average(recall, weights=row_sums)) if row_sums.sum() else 0.0,
            "f1-score": weighted_f1,
            "support": int(row_sums.sum()),
        }
        report_text = pd.DataFrame(report_dict).T.to_string()

    if labels is None:
        labels = ["Down", "Flat", "Up"]
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_text,
        "classification_report_dict": report_dict,
        "confusion_matrix": cm,
        "labels": list(labels),
    }


def evaluate_classifier(model, X_test, y_test, labels=None) -> dict:
    """Predict with a fitted model and evaluate the results."""

    y_pred = model.predict(X_test)
    return evaluate_predictions(y_test, y_pred, labels=labels)
