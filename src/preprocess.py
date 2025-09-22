"""
preprocess.py - load and preprocess sensor CSV for the Smart Factory Agent.

Functions:
- load_data(path): load CSV, parse timestamp column
- preprocess(df): basic cleaning, imputation
""" 
import pandas as pd
import numpy as np
import argparse

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
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fill-method", type=str, default="ffill", help="Option: [median, ffill]")
    parser.add_argument("--input", type=str, default="data/sensors.csv", help="Input CSV path")
    parser.add_argument("--out", type=str, default="data/sensors_processed.csv", help="Output CSV path")
    args = parser.parse_args()

    df = load_data(args.input)
    df = preprocess(df, fill_method=args.fill_method)
    df.to_csv(args.out, index=False)
    print(f"The preprocessed data is saved to {args.out}")

if __name__ == "__main__":
    main()