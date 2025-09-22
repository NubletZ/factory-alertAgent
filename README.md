# Smart Factory Alert Agent - Code Package

This package contains the core code for the Smart Factory Alert Agent described in the assignment. It **does not** include generated dummy sensor data. A template CSV is provided at `data/template.csv` (header-only).

## Structure
- `src/` - Python source files
  - `preprocess.py` - data loading & preprocessing utilities
  - `rule_engine.py` - deterministic rule-based alert and action suggestions
  - `models.py` - ML model wrappers (RandomForest, LightGBM, Logistic Regression supervised baseline)
  - `agent.py` - CLI entrypoint to run the agent on a CSV file
  - `utils.py` - helper functions (save/load, plotting)
- `data/`
  - `template.csv` - CSV header template (no rows)
- `requirements.txt` - Python dependencies

## Quickstart
#### 1. Create a Python environment (Python 3.9+ recommended) and install:
```
$ pip install -r requirements.txt
```

#### 2. The default online-stream CSV at `data/online_stream.csv` (or provide path) with columns: `timestamp,temp,pressure,vibration,label` (label optional for training/eval).

#### 3. Generate some dummy dataset
```
$ python src/generate_data.py --rows 500
```
The arguments that can be set are listed below:
* `--rows` : Number of data point to be generated (100-500)
* `--interval` : Interval in minutes between rows
* `--out` : Output CSV path

#### 4. Preprocess the generated dataset
```
$ python src/preprocess.py
```
There are several arguments that can be set, listed below:
* `--input` : Path to input CSV
* `--out` : Output CSV path
* `--fill-method` : Filling method used in handling missing data, can be `ffill` or `median`.

#### 5. Run the agent in real-time mode:
```
$ python src/agent.py --input data/sensors.csv --output-dir outputs
```
There are several arguments that can be used here:
* `--input` : Path to input CSV
* `--output-dir` : Directory to write outputs
* `--train-if-no-model` : Bool to train model if no model found
* `--verbose` : Integer option: [0, 1, 2], the higher the more logging info

## Notes
- The code is intentionally minimal and dependency-light. A supervised RandomForest trainer is included if you don't want to train from scratch.
- The default datetime interval in generated data is 1 minute, however for demo purpose, we don't really set the data reading interval to be 1 minute.

