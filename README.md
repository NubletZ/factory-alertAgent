# Smart Factory Alert Agent - Code Package

This package contains the core code for the Smart Factory Alert Agent described in the assignment. It **does not** include generated dummy sensor data. A template CSV is provided at `data/template.csv` (header-only).

## Structure
- `src/` - Python source files
  - `preprocess.py` - data loading & preprocessing utilities
  - `rule_engine.py` - deterministic rule-based anomaly detector and suggestions
  - `models.py` - ML model wrappers (IsolationForest + RandomForest supervised baseline)
  - `agent.py` - CLI entrypoint to run the agent on a CSV file
  - `utils.py` - helper functions (save/load, plotting)
- `data/`
  - `template.csv` - CSV header template (no rows)
- `requirements.txt` - Python dependencies

## Quickstart
1. Create a Python environment (Python 3.9+ recommended) and install:
```
pip install -r requirements.txt
```
2. Put your sensor CSV at `data/sensors.csv` (or provide path) with columns:
`timestamp,temp,pressure,vibration,label` (label optional for training/eval).
3. Run the agent in batch mode:
```
python src/agent.py --input data/sensors.csv --output-dir outputs --mode detect
```
This will produce `outputs/anomalies.csv`, `outputs/alerts.json`, and a plot `outputs/signals.png`.

## Notes
- The code is intentionally minimal and dependency-light. The IsolationForest is used as an unsupervised detector. A supervised RandomForest trainer is included if you have `label` column in your CSV.
- No dummy data is included per your request; use your own CSV or populate `data/template.csv`.

