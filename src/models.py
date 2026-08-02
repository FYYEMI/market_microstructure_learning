"""Model training utilities for order book imbalance prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def time_series_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    timestamp_col: str = "timestamp",
):
    """Sort chronologically and split into train/validation/test partitions."""

    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    ordered = df.copy()
    ordered[timestamp_col] = pd.to_datetime(ordered[timestamp_col], utc=True, errors="coerce")
    ordered = ordered.sort_values(timestamp_col).reset_index(drop=True)

    n = len(ordered)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = ordered.iloc[:train_end].copy()
    val_df = ordered.iloc[train_end:val_end].copy()
    test_df = ordered.iloc[val_end:].copy()
    return train_df, val_df, test_df


def get_feature_target_data(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str = "label",
):
    """Return feature matrix and target series without dropping missing feature values."""

    cols = list(feature_cols)
    required = cols + [target_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    subset = df[required].copy()
    subset = subset.dropna(subset=[target_col])
    X = subset[cols]
    y = subset[target_col]
    return X, y


def _preferred_label_order(labels: Iterable[str]) -> list[str]:
    preferred = ["Down", "Flat", "Up"]
    label_set = list(pd.unique(pd.Series(list(labels))))
    ordered = [label for label in preferred if label in label_set]
    ordered.extend([label for label in label_set if label not in ordered])
    return ordered


class NumpyMultinomialLogisticRegression:
    """A small multinomial logistic regression fallback implemented with NumPy."""

    def __init__(self, learning_rate: float = 0.1, max_iter: int = 400, l2: float = 1e-4, random_state: int = 42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2 = l2
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self.median_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def _prepare_X(self, X: pd.DataFrame | np.ndarray, fit: bool = False) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if fit:
            self.median_ = np.nanmedian(X_arr, axis=0)
            X_arr = np.where(np.isnan(X_arr), self.median_, X_arr)
            self.mean_ = X_arr.mean(axis=0)
            self.scale_ = X_arr.std(axis=0)
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        else:
            if self.median_ is None or self.mean_ is None or self.scale_ is None:
                raise RuntimeError("Model must be fit before calling predict.")
            X_arr = np.where(np.isnan(X_arr), self.median_, X_arr)
        X_arr = (X_arr - self.mean_) / self.scale_
        intercept = np.ones((X_arr.shape[0], 1), dtype=float)
        return np.hstack([intercept, X_arr])

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        y_series = pd.Series(y).astype(str)
        self.classes_ = np.array(_preferred_label_order(y_series))
        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = y_series.map(class_to_index).to_numpy()
        X_prepared = self._prepare_X(X, fit=True)
        n_samples, n_features = X_prepared.shape
        n_classes = len(self.classes_)
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(scale=0.01, size=(n_features, n_classes))
        class_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        sample_weights = n_samples / (n_classes * np.maximum(class_counts[y_idx], 1.0))
        y_one_hot = np.eye(n_classes)[y_idx]

        for _ in range(self.max_iter):
            logits = X_prepared @ self.weights_
            probs = self._softmax(logits)
            error = (probs - y_one_hot) * sample_weights[:, None]
            grad = (X_prepared.T @ error) / n_samples
            grad[1:] += self.l2 * self.weights_[1:]
            self.weights_ -= self.learning_rate * grad
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_prepared = self._prepare_X(X, fit=False)
        logits = X_prepared @ self.weights_
        return self._softmax(logits)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train a balanced multinomial logistic regression pipeline."""

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        return model
    except Exception:
        fallback = NumpyMultinomialLogisticRegression(learning_rate=0.1, max_iter=300, l2=1e-4, random_state=42)
        fallback.fit(X_train, y_train)
        return fallback


@dataclass
class XGBoostClassifierWrapper:
    model: object
    label_encoder: LabelEncoder
    feature_cols: list[str]
    imputer: SimpleImputer | None = None

    def predict(self, X):
        X_input = self.imputer.transform(X) if self.imputer is not None else X
        encoded_pred = self.model.predict(X_input)
        return self.label_encoder.inverse_transform(encoded_pred.astype(int))

    def predict_proba(self, X):
        X_input = self.imputer.transform(X) if self.imputer is not None else X
        return self.model.predict_proba(X_input)


def train_xgboost_classifier(X_train: pd.DataFrame, y_train: pd.Series, X_val=None, y_val=None) -> XGBoostClassifierWrapper:
    """Train a simple XGBoost multiclass classifier with label encoding."""

    try:
        from xgboost import XGBClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import LabelEncoder
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("xgboost is required but not available in this environment.") from exc

    feature_cols = list(X_train.columns)
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    num_class = len(label_encoder.classes_)

    classifier = XGBClassifier(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        num_class=num_class,
        n_jobs=1,
    )

    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        X_val_imputed = imputer.transform(X_val)
        y_val_encoded = label_encoder.transform(y_val)
        fit_kwargs = {
            "eval_set": [(X_val_imputed, y_val_encoded)],
            "verbose": False,
        }

    classifier.fit(X_train_imputed, y_train_encoded, **fit_kwargs)
    wrapper = XGBoostClassifierWrapper(model=classifier, label_encoder=label_encoder, feature_cols=feature_cols, imputer=imputer)
    return wrapper
