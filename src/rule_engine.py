"""
rule_engine.py - simple deterministic rules and actionable suggestions.
Thresholds are conservative examples; adjust as needed for real hardware.
"""
from typing import Dict, List, Tuple

# Example thresholds; tune to your plant
THRESHOLDS = {
    "temp": {"low": 43.0, "mid-low": 45, "mid-high": 50, "high": 52.0},
    "pressure": {"low": 0.97, "mid-low": 1.00, "mid-high": 1.05, "high": 1.08},
    "vibration": {"low": 0.02, "mid-high": 0.04, "high": 0.07},
}

SUGGESTIONS = {
    "temp_high": "Temperature too high — check cooling system, verify fan speed and coolant flow.",
    "temp_mid_high": "Temperature a bit high — consider to check cooling system, verify fan speed and coolant flow.",
    "temp_low": "Temperature too low — check heater or process state.",
    "temp_mid_low": "Temperature a bit low — consider to check heater or process state.",
    "pressure_high": "Pressure too high — verify valves and pressure relief; check for blockages.",
    "pressure_mid_high": "Pressure a bit high — consider to verify valves and pressure relief; check for blockages.",
    "pressure_low": "Pressure too low — verify pump operation and supply lines.",
    "pressure_mid_low": "Pressure a bit low — consider to verify pump operation and supply lines.",
    "vibration_high": "Vibration too high — inspect bearings, shaft alignment, and mounting.",
    "vibration_mid_high": "Vibration a bit high — consider to inspect bearings, shaft alignment, and mounting.",
    "vibration_low": "Vibration below normal range — check machine load and drive components; verify sensor mounting."
}

def check_row(row: Dict) -> Tuple[bool, List[str]]:
    alerts = []
    t = row.get("temp", None)
    p = row.get("pressure", None)
    v = row.get("vibration", None)
    if t is not None:
        if t > THRESHOLDS["temp"]["high"]:
            alerts.append(SUGGESTIONS["temp_high"])
            
        elif t > THRESHOLDS["temp"]["mid-high"]:
            alerts.append(SUGGESTIONS["temp_mid_high"])
            
        elif t < THRESHOLDS["temp"]["low"]:
            alerts.append(SUGGESTIONS["temp_low"])
            
        elif t < THRESHOLDS["temp"]["mid-low"]:
            alerts.append(SUGGESTIONS["temp_mid_low"])
            
    if p is not None:
        if p > THRESHOLDS["pressure"]["high"]:
            alerts.append(SUGGESTIONS["pressure_high"])
            
        elif p > THRESHOLDS["pressure"]["mid-high"]:
            alerts.append(SUGGESTIONS["pressure_mid_high"])
            
        elif p < THRESHOLDS["pressure"]["low"]:
            alerts.append(SUGGESTIONS["pressure_low"])
            
        elif p < THRESHOLDS["pressure"]["mid-low"]:
            alerts.append(SUGGESTIONS["pressure_mid_low"])
            
    if v is not None:
        if v > THRESHOLDS["vibration"]["high"]:
            alerts.append(SUGGESTIONS["vibration_high"])
            
        elif v > THRESHOLDS["vibration"]["mid-high"]:
            alerts.append(SUGGESTIONS["vibration_mid_high"])
            
        elif v < THRESHOLDS["vibration"]["low"]:
            alerts.append(SUGGESTIONS["vibration_low"])

    msg_alert = "⚠️  ALERT! Anomaly on machine detected."
    for alert in alerts:
        msg_alert += f"\n - {alert}"
            
    return msg_alert
