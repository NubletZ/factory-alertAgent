"""
agent.py - command-line agent that applies real time anomaly detection.
Usage example:
    python agent.py --input data/sensors.csv --output-dir outputs --mode detect
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import schedule
import time
import os

from preprocess import load_data, preprocess
from rule_engine import check_row
from models import save_model, load_model, score_supervised
from utils import save_csv, save_json, plot_signals
from generate_data import generate_data

MODEL_PATH = "model/randomforest.pth"

def run_detect(input_csv: str, output_dir: str = None, train_if_no_model: bool = True, verbose: int = 1):
    if verbose == 2: print("Read sensor input..")
    df = load_data(input_csv).tail(10)
    
    if verbose == 2: print("Preprocess the data..")
    df = preprocess(df)
    
    if verbose == 2: print(df.tail(3))

    # Prepare features for classifier
    feat_cols = [c for c in df.columns if any(p in c for p in ["temp","pressure","vibration"])]
    X = df[feat_cols]

    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
        except Exception:
            if verbose == 2: print("The model doesn't exist!")
            model = None

    if model is None and train_if_no_model:
        if verbose == 2: print("Train the model..")
        model = train_supervised_rf(X)
        save_model(model, MODEL_PATH)

    if model is not None:
        # Get the latest data point
        cur_data = X.tail(1)
        scores, label = score_supervised(model, cur_data)
        if verbose == 2: 
            print("----------------------------------------------------------------------------")
            print("Prediction result:", label)
            print("Anomaly score:", scores)
            print()
        status = "NORMAL"
        alert_msg = None
        if label != 'normal':
            status = "ABNORMAL!"
            alert_msg = check_row(cur_data.iloc[0].to_dict())
        if verbose > 0: print(f"{np.datetime_as_string(df.timestamp.values[-1], unit='s')} - [{status}] anomaly_score: {scores}, temp: {cur_data.temp.values[0]}, pressure: {cur_data.pressure.values[0]}, vibration: {cur_data.vibration.values[0]}")
        if alert_msg != None:
            print(alert_msg)
            print()
        if verbose == 2: 
            print("============================================================================")

def simulate_sensor_data(input_csv: str):
    # Simulate sensor data generation and append to CSV
    # Generate new data
    file_exists = os.path.exists(input_csv)
    prev_data = None
    if file_exists:
        prev_data = pd.read_csv(input_csv).tail(10)

    df_to_write = generate_data(rows=1, prev_data=prev_data)

    # Normalize timestamp to a consistent string format if possible
    try:
        df_to_write["timestamp"] = pd.to_datetime(df_to_write["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        # fallback: cast to string
        df_to_write["timestamp"] = df_to_write["timestamp"].astype(str)

    # Append if file exists, otherwise create with header
    if os.path.exists(input_csv):
        df_to_write.to_csv(input_csv, mode="a+", header=False, index=False)
    else:
        df_to_write.to_csv(input_csv, index=False)


# Schedule the function to run every minute
def run_cycle(input_csv: str, output_dir: str = None, train_if_no_model: bool = True):
    # run multiple tasks in order; wrap in try/except so one failure doesn't stop subsequent tasks
    try:
        simulate_sensor_data(input_csv)
    except Exception as e:
        print("simulate_sensor_data failed:", e)
    try:
        run_detect(input_csv)
    except Exception as e:
        print("run_detect failed:", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='data/online_stream.csv', help="Path to input CSV")
    parser.add_argument("--output-dir", default=None, help="Directory to write outputs")
    parser.add_argument("--train-if-no-model", default=True, help="Bool to train model if no model found")
    parser.add_argument("--verbose", default=1, help="Integer option: [0, 1, 2], the higher the more logging info")
    args = parser.parse_args()
    schedule.every(0.1).minutes.do(run_cycle, input_csv=args.input, output_dir=args.output_dir, train_if_no_model=args.train_if_no_model)
    print("Agent started. Running receive sensor input -> detect anomaly every 1 minute...")

    while True:
        schedule.run_pending()
        time.sleep(0.1)

if __name__ == "__main__":
    main()