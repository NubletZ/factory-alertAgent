"""
agent.py - command-line agent that applies rule-based checks and an IsolationForest detector.
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
from models import train_isolation_forest, score_isolation_forest, save_model, load_model, score_supervised_rf
from utils import save_csv, save_json, plot_signals
from generate_data import generate_data

MODEL_PATH = "model/randomforest.pth"

def run_detect(input_csv: str, output_dir: str = None, train_if_no_model: bool = True):
    # print("Load CSV data..")
    df = load_data(input_csv).tail(10)
    # if "timestamp" not in df.columns:
    #     raise ValueError("Input CSV must contain a 'timestamp' column.")
    # print("Preprocess the data..")
    df = preprocess(df)
    # Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Rule-based checks
    # alerts = []
    # for i, row in df.iterrows():
    #     is_anom, suggestions = check_row(row)
    #     if is_anom:
    #         alerts.append({
    #             "timestamp": str(row["timestamp"]),
    #             "temp": row.get("temp"),
    #             "pressure": row.get("pressure"),
    #             "vibration": row.get("vibration"),
    #             "source": "rule",
    #             "suggestions": suggestions
    #         })

    # Prepare features for IsolationForest
    feat_cols = [c for c in df.columns if any(p in c for p in ["temp","pressure","vibration"])]
    X = df[feat_cols]

    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
        except Exception:
            print("The model doesn't exist!")
            model = None

    if model is None and train_if_no_model:
        print("Train the model")
        model = train_supervised_rf(X)
        save_model(model, MODEL_PATH)

    if model is not None:
        cur_data = X.tail(1)
        scores, label = score_supervised_rf(model, cur_data)
        status = "NORMAL"
        alert_msg = None
        if label != 'normal':
            status = "ABNORMAL!"
            alert_msg = check_row(cur_data.iloc[0].to_dict())
        print(f"{np.datetime_as_string(df.timestamp.values[-1], unit='s')} - [{status}] anomaly_score: {scores}, temp: {cur_data.temp.values[0]}, pressure: {cur_data.pressure.values[0]}, vibration: {cur_data.vibration.values[0]}")
        if alert_msg != None:
            print(alert_msg)
            print()

def simulate_sensor_data(input_csv: str):
    # Simulate sensor data generation and append to CSV
    # Generate new data
    file_exists = os.path.exists(input_csv)
    prev_data = None
    if file_exists:
        prev_data = pd.read_csv(input_csv).tail(10)

    df_to_write = generate_data(rows=1, prev_data=prev_data)

    # # Check if the CSV exists, append or create accordingly
    # cols = ["timestamp", "temp", "pressure", "vibration", "label"]

    # # Ensure new_df has the required columns and order
    # df_to_write = new_df.copy()
    # for c in cols:
    #     if c not in df_to_write.columns:
    #         df_to_write[c] = ""

    # df_to_write = df_to_write[cols]

    # Normalize timestamp to a consistent string format if possible
    try:
        df_to_write["timestamp"] = pd.to_datetime(df_to_write["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        # fallback: cast to string
        df_to_write["timestamp"] = df_to_write["timestamp"].astype(str)

    # Append if file exists, otherwise create with header
    if os.path.exists(csv_path):
        df_to_write.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_to_write.to_csv(csv_path, index=False)


# Schedule the function to run every minute
def run_cycle(input_csv: str, output_dir: str = None, train_if_no_model: bool = True):
    # run multiple tasks in order; wrap in try/except so one failure doesn't stop subsequent tasks
    try:
        simulate_sensor_data(csv_path)
    except Exception as e:
        print("simulate_sensor_data failed:", e)
    try:
        run_detect(csv_path)
    except Exception as e:
        print("run_detect failed:", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write outputs")
    # parser.add_argument("--mode", choices=["detect"], default="detect")
    args = parser.parse_args()
    # if args.mode == "detect":
    #     run_detect(args.input, args.output_dir)

if __name__ == "__main__":
    main()


# Path to the CSV file
csv_path = "C:/temp/Pegatron/test/smart-factory-agent/data/sensors_processed2.csv"

# schedule the ordered cycle (adjust interval as needed)
schedule.every(0.1).minutes.do(run_cycle)
print("Agent started. Running simulate_sensor_data -> run_detect every 1 minute...")

while True:
    schedule.run_pending()
    time.sleep(0.1)