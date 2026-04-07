import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


def load_data(path):
    """Load a CSV dataset from disk."""
    return pd.read_csv(path)


def preprocess_data(df, target_col="loan_approval", sensitive_col=None):
    """
    Full preprocessing pipeline:
    - Separate features and target
    - Preserve raw sensitive features for fairness metrics
    - Impute missing values (median for numeric, mode for categorical)
    - Label-encode categorical columns
    - Standardize numeric columns
    """
    df = df.copy()

    y = df[target_col]
    X = df.drop(columns=[target_col])

    sensitive_features_raw = None
    if sensitive_col and sensitive_col in df.columns:
        sensitive_features_raw = df[sensitive_col].copy()

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        X[num_cols] = num_imputer.fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    sensitive_features = None
    if sensitive_col and sensitive_col in X.columns:
        sensitive_features = X[sensitive_col].copy()

    if len(num_cols) > 0:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y, sensitive_features, encoders, sensitive_features_raw


def get_data_profile(df):
    """
    Generate a data profiling summary for the Data Management page.
    Returns a dict with row_count, col_count, missing_summary, dtypes_summary.
    """
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    profile = {
        "row_count": len(df),
        "col_count": len(df.columns),
        "missing_counts": missing.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "total_missing": int(missing.sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_cols": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_cols": df.select_dtypes(exclude=["number"]).columns.tolist(),
    }
    return profile
