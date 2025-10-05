import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# --- Paths -------------------------------------------------------------
# Resolve project root from this file's location (works on Windows/macOS/Linux)
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "data", "census.csv")
MODEL_DIR = os.path.join(PROJECT_PATH, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading data from: {DATA_PATH}")
data = pd.read_csv(DATA_PATH)

# --- Split -------------------------------------------------------------
# Keep label distribution similar in both splits
train, test = train_test_split(
    data, test_size=0.20, random_state=42, stratify=data["salary"]
)

# --- Categorical features (match CSV column names exactly) -------------
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# --- Process data ------------------------------------------------------
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,     # fit encoder & label binarizer
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,    # use fitted encoder & lb
    encoder=encoder,
    lb=lb,
)

# --- Train -------------------------------------------------------------
model = train_model(X_train, y_train)

# --- Save artifacts ----------------------------------------------------
model_path = os.path.join(MODEL_DIR, "model.pkl")
encoder_path = os.path.join(MODEL_DIR, "encoder.pkl")
lb_path = os.path.join(MODEL_DIR, "lb.pkl")  # helpful later for API

save_model(model, model_path)
save_model(encoder, encoder_path)
save_model(lb, lb_path)

print(f"Model saved to:   {model_path}")
print(f"Encoder saved to: {encoder_path}")
print(f"LabelBinarizer saved to: {lb_path}")

# --- Load model back (sanity check) -----------------------------------
model = load_model(model_path)

# --- Inference & metrics ----------------------------------------------
preds = inference(model, X_test)

p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# --- Slice performance -------------------------------------------------
# Reset (overwrite) the slice output file once, then append inside loop
slice_out_path = os.path.join(PROJECT_PATH, "slice_output.txt")
open(slice_out_path, "w").close()

for col in cat_features:
    # iterate through unique values of this categorical column
    for slice_value in sorted(test[col].dropna().unique()):
        count = test[test[col] == slice_value].shape[0]
        sp, sr, sfb = performance_on_categorical_slice(
            data=test,
            column_name=col,
            slice_value=slice_value,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model,
        )
        with open(slice_out_path, "a", encoding="utf-8") as f:
            print(f"{col}: {slice_value}, Count: {count:,}", file=f)
            print(f"Precision: {sp:.4f} | Recall: {sr:.4f} | F1: {sfb:.4f}", file=f)

print(f"Slice metrics written to: {slice_out_path}")
