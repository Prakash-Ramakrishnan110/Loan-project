import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df, target_col='loan_approval', sensitive_col=None):
    df = df.copy()
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    sensitive_features_raw = None
    if sensitive_col and sensitive_col in df.columns:
        sensitive_features_raw = df[sensitive_col]
        
    num_cols = X.select_dtypes(include=['number']).columns
    cat_cols = X.select_dtypes(exclude=['number']).columns
    
    # Missing values
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X[num_cols] = num_imputer.fit_transform(X[num_cols])
        
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        
    # Encodings
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        
    # Sensitive features processed
    sensitive_features = None
    if sensitive_col and sensitive_col in X.columns:
        sensitive_features = X[sensitive_col]
        
    # Normalize
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        
    return X, y, sensitive_features, encoders, sensitive_features_raw
