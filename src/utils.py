"""
utils.py - simple helpers for saving outputs and plotting.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def save_json(obj, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_csv(df: pd.DataFrame, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

def plot_signals(df, cols=["temp","pressure","vibration"], outpath="outputs/signals.png"):
    p = Path(outpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,6))
    for c in cols:
        if c in df.columns:
            plt.plot(df["timestamp"], df[c], label=c)
    plt.legend()
    plt.xlabel("timestamp")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
