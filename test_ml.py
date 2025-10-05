# test_ml.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Keep column names EXACTLY as in census.csv (hyphens included)
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


def _toy_df() -> pd.DataFrame:
    """Small synthetic dataset with the same schema as census.csv."""
    rows = [
        {
            "age": 39, "workclass": "State-gov", "fnlgt": 77516,
            "education": "Bachelors", "education-num": 13,
            "marital-status": "Never-married", "occupation": "Adm-clerical",
            "relationship": "Not-in-family", "race": "White", "sex": "Male",
            "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40,
            "native-country": "United-States", "salary": "<=50K",
        },
        {
            "age": 50, "workclass": "Self-emp-not-inc", "fnlgt": 83311,
            "education": "Bachelors", "education-num": 13,
            "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
            "relationship": "Husband", "race": "White", "sex": "Male",
            "capital-gain": 0, "capital-loss": 0, "hours-per-week": 60,
            "native-country": "United-States", "salary": ">50K",
        },
        {
            "age": 28, "workclass": "Private", "fnlgt": 338409,
            "education": "Bachelors", "education-num": 13,
            "marital-status": "Married-civ-spouse", "occupation": "Prof-specialty",
            "relationship": "Wife", "race": "Black", "sex": "Female",
            "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40,
            "native-country": "United-States", "salary": ">50K",
        },
        {
            "age": 38, "workclass": "Private", "fnlgt": 215646,
            "education": "HS-grad", "education-num": 9,
            "marital-status": "Divorced", "occupation": "Handlers-cleaners",
            "relationship": "Not-in-family", "race": "White", "sex": "Male",
            "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40,
            "native-country": "United-States", "salary": "<=50K",
        },
        {
            "age": 37, "workclass": "Private", "fnlgt": 284582,
            "education": "Masters", "education-num": 14,
            "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
            "relationship": "Husband", "race": "White", "sex": "Male",
            "capital-gain": 0, "capital-loss": 0, "hours-per-week": 60,
            "native-country": "United-States", "salary": ">50K",
        },
        {
            "age": 53, "workclass": "Private", "fnlgt": 234721,
            "education": "11th", "education-num": 7,
            "marital-status": "Married-civ-spouse", "occupation": "Handlers-cleaners",
            "relationship": "Husband", "race": "Black", "sex": "Male",
            "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40,
            "native-country": "United-States", "salary": "<=50K",
        },
    ]
    return pd.DataFrame(rows)


def test_process_data_outputs():
    """
    process_data should return numpy arrays of matching length and non-null encoder/lb.
    """
    df = _toy_df()
    X, y, enc, lb = process_data(
        df, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0] == df.shape[0]
    assert enc is not None and lb is not None


def test_train_model_and_inference():
    """
    train_model should return a RandomForestClassifier and inference should output 0/1 labels
    of the correct length.
    """
    df = _toy_df()
    X, y, _, _ = process_data(
        df, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_range():
    """
    compute_model_metrics should return precision/recall/F1 within [0, 1].
    """
    df = _toy_df()
    X, y, _, _ = process_data(
        df, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    p, r, f1 = compute_model_metrics(y, preds)
    for m in (p, r, f1):
        assert 0.0 <= m <= 1.0
