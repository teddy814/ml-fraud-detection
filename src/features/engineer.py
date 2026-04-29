"""Feature engineering for fraud detection."""
import numpy as np
import pandas as pd
 
 
def add_time_features(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """
    Extract temporal features from raw elapsed-seconds timestamp.
    These help the model detect time-of-day fraud patterns.
    """
    df["hour"] = (df[time_col] // 3600) % 24
    df["day"] = (df[time_col] // 86400) % 7
    # Is this a night-time transaction? (midnight–6am is higher risk)
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    return df
 
 
def add_amount_features(df: pd.DataFrame, amount_col: str = "Amount") -> pd.DataFrame:
    """
    Transform and bin transaction amounts.
    Raw Amount is heavily right-skewed; log transform improves model performance.
    """
    df["log_amount"] = np.log1p(df[amount_col])
    df["amount_bin"] = pd.qcut(
        df[amount_col], q=10, labels=False, duplicates="drop"
    )
    # Flag round-number amounts (e.g. £100.00) — common in fraud
    df["is_round_amount"] = (df[amount_col] % 1 == 0).astype(int)
    return df
 
 
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple interaction terms between key PCA features and amount.
    Helps capture non-linear relationships the model might miss.
    """
    for v in ["V1", "V2", "V3", "V4", "V10", "V11", "V12", "V14", "V17"]:
        if v in df.columns:
            df[f"{v}_x_log_amount"] = df[v] * df["log_amount"]
    return df
 
 
def get_feature_cols(df: pd.DataFrame, target: str = "Class") -> list:
    """Return all feature columns (everything except the target)."""
    return [c for c in df.columns if c != target]
 