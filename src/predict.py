"""
Run inference on new customer data using the saved Random Forest model.

Usage
-----
    python3 src/predict.py --input data/new_customers.csv
    python3 src/predict.py --input data/new_customers.csv --model lr
"""

import argparse
import os

import joblib
import pandas as pd

from src.config import LR_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH
from src.preprocess import _cap_outliers, _clean, _encode


def predict(input_csv: str, model_choice: str = "rf", output_csv: str = "reports/predictions.csv") -> pd.DataFrame:
    """
    Load a new CSV, preprocess, and return subscription predictions.

    Args:
        input_csv:    Path to raw customer data CSV (same schema as training data).
        model_choice: 'rf' (Random Forest) or 'lr' (Logistic Regression).
        output_csv:   Where to save predictions.

    Returns:
        DataFrame with a 'predicted_subscription' column (0 = no, 1 = yes).
    """
    df_raw = pd.read_csv(input_csv)
    df = _clean(df_raw)
    df = _cap_outliers(df)
    df = _encode(df)

    # Drop target if present (won't be for new data)
    if "term_deposit" in df.columns:
        df.drop(columns=["term_deposit"], inplace=True)

    # Apply fitted scaler
    from src.config import SCALE_COLS, LOW_INFO_FEATURES
    drop_cols = [c for c in LOW_INFO_FEATURES if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    scaler = joblib.load(SCALER_PATH)
    scale_cols = [c for c in SCALE_COLS if c in df.columns]
    df[scale_cols] = scaler.transform(df[scale_cols])

    # Load model and predict
    model_path = RF_MODEL_PATH if model_choice == "rf" else LR_MODEL_PATH
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None

    result = pd.DataFrame({"predicted_subscription": preds})
    if probs is not None:
        result["probability"] = probs.round(4)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Predictions saved → {output_csv}")
    print(f"Prospective customers: {preds.sum():,} / {len(preds):,} ({preds.mean()*100:.1f}%)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict prospective customers.")
    parser.add_argument("--input",  required=True, help="Path to new customer CSV")
    parser.add_argument("--model",  default="rf", choices=["rf", "lr"],
                        help="Model to use: 'rf' or 'lr' (default: rf)")
    parser.add_argument("--output", default="reports/predictions.csv",
                        help="Output CSV path (default: reports/predictions.csv)")
    args = parser.parse_args()
    predict(args.input, args.model, args.output)
