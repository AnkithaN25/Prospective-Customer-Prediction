"""
Data loading, cleaning, encoding, outlier handling, and scaling.

Usage
-----
    from src.preprocess import load_and_preprocess

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    BASIC_EDUCATION_LABEL,
    BASIC_EDUCATION_VARIANTS,
    BINARY_MAP,
    DATA_PATH,
    DAY_MAP,
    LOW_INFO_FEATURES,
    MONTH_MAP,
    OUTLIER_CAPS,
    RANDOM_STATE,
    SCALE_COLS,
    SCALER_PATH,
    TARGET,
    TARGET_MAP,
    TEST_SIZE,
)


# ── Individual steps ──────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names, consolidate education labels, drop duplicates."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Consolidate all basic education variants
    df.loc[df["education"].isin(BASIC_EDUCATION_VARIANTS), "education"] = BASIC_EDUCATION_LABEL

    # Create binary prev_c: was customer contacted in a previous campaign?
    df["prev_c"] = df["pdays"].apply(lambda x: "no" if x == 999 else "yes")
    df.drop(columns=["pdays"], inplace=True)

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate rows. Remaining: {len(df):,}")
    return df


def _cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers at IQR-derived upper bounds (winsorising)."""
    df = df.copy()
    for col, cap in OUTLIER_CAPS.items():
        df.loc[df[col] > cap, col] = cap
    return df


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all categorical columns:
        - month / day_of_week → integer (ordinal)
        - default / housing / loan → {yes:1, no:0, unknown:-1}
        - term_deposit / prev_c → {yes:1, no:0}
        - contact / poutcome / job / education / marital → one-hot (drop_first=True)
        - 'unknown' variants renamed to avoid merge with other columns
    """
    df = df.copy()

    # Rename 'unknown' to distinguish per-column before one-hot encoding
    for col, suffix in [("job", "j"), ("education", "e"), ("marital", "m")]:
        df.loc[df[col] == "unknown", col] = f"unknown{suffix}"

    df["month"]       = df["month"].map(MONTH_MAP)
    df["day_of_week"] = df["day_of_week"].map(DAY_MAP)

    for col in ("default", "housing", "loan"):
        df[col] = df[col].map(BINARY_MAP)

    df[TARGET]   = df[TARGET].map(TARGET_MAP)
    df["prev_c"] = df["prev_c"].map(TARGET_MAP)

    ohe_cols = ["contact", "poutcome", "job", "education", "marital"]
    dummies  = pd.get_dummies(df[ohe_cols], prefix="d", drop_first=True)
    df = pd.concat([df.drop(columns=ohe_cols), dummies], axis=1)

    return df


def _scale(df: pd.DataFrame, fit: bool = True, scaler=None):
    """
    StandardScaler on continuous numeric columns only (not dummies/binary).

    Args:
        df:     Encoded DataFrame.
        fit:    If True, fit a new scaler and return it. If False, apply provided scaler.
        scaler: Pre-fitted scaler (required when fit=False).

    Returns:
        (scaled_df, scaler)
    """
    cols_to_scale = [c for c in SCALE_COLS if c in df.columns]

    if fit:
        scaler = StandardScaler()
        scaler.fit(df[cols_to_scale])

    df = df.copy()
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df, scaler


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_preprocess(
    data_path: str = DATA_PATH,
    save_scaler: bool = True,
) -> tuple:
    """
    Full pipeline: load → clean → cap outliers → encode → scale → split.

    Args:
        data_path:   Path to the raw CSV.
        save_scaler: Persist fitted scaler to SCALER_PATH.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    df = _clean(df)
    df = _cap_outliers(df)
    df = _encode(df)

    # Drop low mutual-information features identified during EDA
    drop_cols = [c for c in LOW_INFO_FEATURES if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    df, scaler = _scale(df, fit=True)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    if save_scaler:
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"Scaler saved → {SCALER_PATH}")

    return X_train, X_test, y_train, y_test, scaler
