import pickle
from typing import Any, Iterable, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


# --- Train -------------------------------------------------------------

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Trains a machine learning model and returns it.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


# --- Metrics -----------------------------------------------------------

def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.
    Returns (precision, recall, fbeta).
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


# --- Inference ---------------------------------------------------------

def inference(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Run model inferences and return the predictions (0/1).
    """
    return model.predict(X)


# --- Persistence -------------------------------------------------------

def save_model(obj: Any, path: str) -> None:
    """
    Serializes a model or encoder object to a pickle file.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_model(path: str) -> Any:
    """
    Loads a pickle file from `path` and returns the object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# --- Slice Performance -------------------------------------------------

def performance_on_categorical_slice(
    data: pd.DataFrame,
    column_name: str,
    slice_value: Any,
    categorical_features: Iterable[str],
    label: str,
    encoder,
    lb,
    model: Any
) -> Tuple[float, float, float]:
    """
    Computes (precision, recall, f1) on the subset of `data` where
    `column_name == slice_value`, using the provided encoder/lb/model.

    Notes:
    - Uses process_data(training=False, encoder=..., lb=...) to ensure
      consistent transformations with training.
    - Returns (0,0,0) if the slice has no rows.
    """
    # Filter to the requested slice
    df_slice = data[data[column_name] == slice_value].copy()
    if df_slice.empty:
        return 0.0, 0.0, 0.0

    # Transform slice using existing encoder/label binarizer
    X_slice, y_slice, _, _ = process_data(
        df_slice,
        categorical_features=list(categorical_features),
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Predict & score
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
