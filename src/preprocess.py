"""
preprocess.py - load and preprocess sensor CSV for the Smart Factory Agent.

Functions:
- load_data(path): load CSV, parse timestamp column
- preprocess(df): basic cleaning, imputation, scaling and feature engineering
""" 
import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def preprocess(df: pd.DataFrame, fill_method: str = "ffill", window: int = 5) -> pd.DataFrame:
    df = df.copy()
    # Ensure numeric types
    for col in ["temp", "pressure", "vibration"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Simple imputation
    if fill_method == "ffill":
        df[["temp", "pressure", "vibration"]] = df[["temp", "pressure", "vibration"]].ffill().bfill()
    elif fill_method == "median":
        df[["temp","pressure","vibration"]] = df[["temp","pressure","vibration"]].fillna(df[["temp","pressure","vibration"]].median())
    # Feature engineering: rolling mean/std and z-score
    # for col in ["temp","pressure","vibration"]:
    #     if col in df.columns:
    #         df[f"{col}_rm"] = df[col].rolling(window=window, min_periods=1).mean()
    #         df[f"{col}_rs"] = df[col].rolling(window=window, min_periods=1).std().fillna(0.0)
    #         # z-score (using rolling mean/std)
    #         df[f"{col}_z"] = (df[col] - df[f"{col}_rm"]) / (df[f"{col}_rs"].replace({0: np.nan}))
    #         df[f"{col}_z"] = df[f"{col}_z"].fillna(0.0)
    return df


# TEST
df = load_data("C:/temp/Pegatron/test/smart-factory-agent/data/sensors.csv")
df = preprocess(df, fill_method="median")
df.to_csv("data/sensors_processed.csv", index=False)