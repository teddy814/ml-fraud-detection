"""Data loading and initial preprocessing."""
import pandas as pd
from pathlib import Path
 
DATA_RAW = Path(__file__).parents[2] / "data" / "raw"
DATA_PROCESSED = Path(__file__).parents[2] / "data" / "processed"
 
REQUIRED_COLS = {"Time", "Amount", "Class"}
EXPECTED_CLASSES = {0, 1}
 
 
def load_raw(filename: str) -> pd.DataFrame:
    """Load raw CSV data with basic validation."""
    path = DATA_RAW / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Download creditcard.csv from:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it in data/raw/"
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns from {filename}")
    _validate(df)
    return df
 
 
def _validate(df: pd.DataFrame) -> None:
    """Raise informative errors for common data issues."""
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    unexpected_classes = set(df["Class"].unique()) - EXPECTED_CLASSES
    if unexpected_classes:
        raise ValueError(f"Unexpected class values: {unexpected_classes}")
    fraud_rate = df["Class"].mean()
    print(f"  Fraud rate: {fraud_rate:.3%}  ({df['Class'].sum():,} fraud / {len(df):,} total)")
 
 
def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Save processed dataframe to parquet (faster and typed vs CSV)."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / filename
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} rows to {filename}")
 
 
def load_processed(filename: str) -> pd.DataFrame:
    """Load previously processed parquet file."""
    path = DATA_PROCESSED / filename
    return pd.read_parquet(path)
 