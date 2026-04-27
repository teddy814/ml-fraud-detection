"""Cleaning and preprocessing pipeline."""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
 
MODELS_DIR = Path(__file__).parents[2] / "models"
 
 
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with column medians.
    The credit card dataset rarely has nulls, but this guards against edge cases.
    """
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    missing = df.isnull().sum().sum()
    print(f"Missing values after imputation: {missing}")
    return df
 
 
def scale_features(df: pd.DataFrame, feature_cols: list, scaler=None):
    """
    Standardise numeric features using StandardScaler.
 
    If scaler is None, fit a new one (training). Otherwise apply existing (inference).
    Returns (df, fitted_scaler).
    """
    if scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        print(f"Fitted new scaler on {len(feature_cols)} features.")
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
        print("Applied existing scaler.")
    return df, scaler
 
 
def save_scaler(scaler: StandardScaler, name: str = "scaler") -> None:
    """Persist the fitted scaler so inference uses the same scaling."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / f"{name}.pkl")
    print(f"Scaler saved → models/{name}.pkl")
 
 
def load_scaler(name: str = "scaler") -> StandardScaler:
    return joblib.load(MODELS_DIR / f"{name}.pkl")
 