"""
generate_data.py - create dummy sensor dataset for Smart Factory Agent.

Usage:
    python src/generate_data.py --rows 300 --interval 1 --out data/sensors.csv

Args:
    --rows: number of rows (100-500 recommended)
    --interval: time step in minutes (default: 1)
    --out: output CSV path
"""

import argparse
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Normal ranges
NORMAL_RANGES = {
    "temp": (0.45, 0.50),        # °C
    "pressure": (1.00, 1.05),    # bar
    "vibration": (0.02, 0.04),   # g
}

# Abnormal thresholds (values outside are anomalies)
ABNORMAL_THRESHOLDS = {
    "temp": (0.43, 0.52),
    "pressure": (0.97, 1.08),
    "vibration": (0.0, 0.07),
}

# Scale anomaly injection range
SCALE_RANGES = 0.1

def generate_data_point(prev_data, dtype):
    if np.random.rand() < 0.1:
        # anomaly injection
        anomaly_ts = ABNORMAL_THRESHOLDS[dtype]
        if prev_data - anomaly_ts[0] > anomaly_ts[1] - prev_data:
            candidate = np.random.uniform(anomaly_ts[0] - SCALE_RANGES, anomaly_ts[0])
        else:
            candidate = np.random.uniform(anomaly_ts[1], anomaly_ts[1] + SCALE_RANGES)
    else:
        # candidate sample (normal or anomaly depending on random draw)
        candidate = np.random.uniform(*NORMAL_RANGES[dtype])

    # apply smoothing formula
    final_data = (2 * candidate + prev_data) / 3.0
    
    if dtype == "temp":
        return final_data * 100
    return final_data

def data_labeling(temp, pressure, vibration):
    state = "normal"
    if temp < NORMAL_RANGES["temp"][0] or temp > NORMAL_RANGES["temp"][1]:
        state = "abnormal"
    elif pressure < NORMAL_RANGES["pressure"][0] or pressure > NORMAL_RANGES["pressure"][1]:
        state = "abnormal"
    elif vibration < NORMAL_RANGES["vibration"][0] or vibration > NORMAL_RANGES["vibration"][1]:
        state = "abnormal"
    return state

def generate_data(rows=200, interval=1, start_time=None, seed=42, prev_data=None):
    if prev_data is not None:
        prev_data = prev_data.ffill().bfill()
        timestamps = [ pd.to_datetime(prev_data['timestamp']).iloc[-1].to_pydatetime() + timedelta(minutes=i*interval) for i in range(1,rows+1)]
        temp = prev_data['temp'].values[-1]
        pressure = prev_data['pressure'].values[-1]
        vibration = prev_data['vibration'].values[-1]
    else:
        np.random.seed(seed)
        if start_time is None:
            start_time = datetime.now().replace(second=0, microsecond=0)

        timestamps = [start_time + timedelta(minutes=i*interval) for i in range(rows)]

        # Start with normal values
        temp = np.random.uniform(*NORMAL_RANGES["temp"])
        pressure = np.random.uniform(*NORMAL_RANGES["pressure"])
        vibration = np.random.uniform(*NORMAL_RANGES["vibration"])

    data = []
    for t in timestamps:
        temp = generate_data_point(temp/100, 'temp')
        pressure = generate_data_point(pressure, 'pressure')
        vibration = generate_data_point(vibration, 'vibration')

        # Add some Gaussian noise to simulate real signals
        temp += np.random.normal(0, 0.2)
        pressure += np.random.normal(0, 0.005)
        vibration += np.random.normal(0, 0.002)

        label = data_labeling(temp/100, pressure, vibration)

        cur_data = [t.isoformat(), round(temp, 2), round(pressure, 3), round(vibration, 3), label]

        # Delete some data to simulate mising data
        if np.random.rand() < 0.05:
            idx = random.sample(range(1,4), 1)[0]
            cur_data[idx] = None

        data.append(cur_data)

    return pd.DataFrame(data, columns=["timestamp", "temp", "pressure", "vibration", "label"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500, help="Number of rows (100-500)")
    parser.add_argument("--interval", type=int, default=1, help="Interval in minutes between rows")
    parser.add_argument("--out", type=str, default="data/sensors.csv", help="Output CSV path")
    args = parser.parse_args()

    df = generate_data(rows=args.rows, interval=args.interval)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} rows at {args.out}")


if __name__ == "__main__":
    main()
