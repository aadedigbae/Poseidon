"""
storage_local.py

Local file-based storage backend for Poseidon.
Fixed for src/ directory structure.

Data layout:
root/
  data/
    storage/
      ponds/
        RWA-01/
          readings.csv
          risk_predictions.csv
          forecasts.csv
          actions_log.csv
"""

from pathlib import Path
from typing import Dict, Any, List
import os
import csv
import pandas as pd
from datetime import datetime

# For files in src/, go UP to reach root, then data/
SCRIPT_DIR = Path(__file__).resolve().parent  # This is src/
ROOT = SCRIPT_DIR.parent  # This is root/ (one level up)
DATA = ROOT / "data"
STORAGE_ROOT = DATA / "storage" / "ponds"


def ensure_pond_folder(pond_id: str) -> Path:
    """
    Ensure the pond-specific folder exists.
    Example: root/data/storage/ponds/RWA-01/
    """
    pond_dir = STORAGE_ROOT / pond_id
    pond_dir.mkdir(parents=True, exist_ok=True)
    return pond_dir


def _append_csv_row(csv_path: Path, row: Dict[str, Any], field_order: List[str] = None):
    """
    Append a single dictionary row to a CSV.
    If the file doesn't exist, write a header first.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    # Normalize timestamp objects to ISO strings for CSV
    clean_row = {}
    for k, v in row.items():
        if isinstance(v, datetime):
            clean_row[k] = v.isoformat()
        else:
            clean_row[k] = v

    if not file_exists:
        # First time: define header
        if field_order is None:
            field_order = sorted(clean_row.keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=field_order)
            writer.writeheader()
            writer.writerow(clean_row)
    else:
        # Append, but ensure consistent columns
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader)
        # Align row to existing header
        aligned_row = {h: clean_row.get(h, "") for h in existing_header}
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=existing_header)
            writer.writerow(aligned_row)


# Public API
def save_reading(pond_id: str, reading: Dict[str, Any]) -> None:
    """Save a single sensor reading into readings.csv for this pond."""
    pond_dir = ensure_pond_folder(pond_id)
    csv_path = pond_dir / "readings.csv"
    _append_csv_row(csv_path, reading)


def get_recent_readings(pond_id: str, n_steps: int) -> pd.DataFrame:
    """
    Return the last n_steps readings for this pond as a DataFrame,
    ordered by timestamp ascending.
    """
    pond_dir = ensure_pond_folder(pond_id)
    csv_path = pond_dir / "readings.csv"

    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    if len(df) > n_steps:
        df = df.iloc[-n_steps:]

    df = df.reset_index(drop=True)
    return df


def save_risk_prediction(pond_id: str, risk: Dict[str, Any]) -> None:
    """Save one risk prediction row in risk_predictions.csv."""
    pond_dir = ensure_pond_folder(pond_id)
    csv_path = pond_dir / "risk_predictions.csv"
    _append_csv_row(csv_path, risk)


def save_forecast(pond_id: str, forecast: Dict[str, Any]) -> None:
    """Save one forecast row in forecasts.csv."""
    pond_dir = ensure_pond_folder(pond_id)
    csv_path = pond_dir / "forecasts.csv"
    _append_csv_row(csv_path, forecast)


def save_actions(pond_id: str, actions: Dict[str, Any]) -> None:
    """Save a row into actions_log.csv."""
    pond_dir = ensure_pond_folder(pond_id)
    csv_path = pond_dir / "actions_log.csv"
    _append_csv_row(csv_path, actions)


'''
"""
storage_local.py

Local file-based storage backend for Poseidon.
Data layout:

Poseidon/
  data/
    storage/
      ponds/
        RWA-01/
          readings.csv
          risk_predictions.csv
          forecasts.csv
          actions_log.csv

This is useful for:
- local development
- offline experiments
- easy inspection with pandas / Excel

You can later replace this with a Firebase backend
without changing the rest of your code.
"""

from pathlib import Path
from typing import Dict, Any, List

import os
import csv
import pandas as pd
from datetime import datetime

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]   # Poseidon/
DATA = ROOT / "data"
STORAGE_ROOT = DATA / "storage" / "ponds"


def ensure_pond_folder(pond_id: str) -> Path:
    """
    Ensure the pond-specific folder exists.
    Example: data/storage/ponds/RWA-01/
    """
    pond_dir = STORAGE_ROOT / pond_id
    pond_dir.mkdir(parents=True, exist_ok=True)
    return pond_dir


def _append_csv_row(csv_path: Path, row: Dict[str, Any], field_order: List[str] = None):
    """
    Append a single dictionary row to a CSV.
    If the file doesn't exist, write a header first.

    field_order: optional explicit column order.
    If None, columns will follow sorted(row.keys()) for new file.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    # Normalize timestamp objects to ISO strings for CSV
    clean_row = {}
    for k, v in row.items():
        if isinstance(v, datetime):
            clean_row[k] = v.isoformat()
        else:
            clean_row[k] = v

    if not file_exists:
        # First time: define header
        if field_order is None:
            field_order = sorted(clean_row.keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=field_order)
            writer.writeheader()
            writer.writerow(clean_row)
    else:
        # Append, but ensure consistent columns
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader)
        # Align row to existing header
        aligned_row = {h: clean_row.get(h, "") for h in existing_header}
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=existing_header)
            writer.writerow(aligned_row)


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def save_reading(pond_id: str, reading: Dict[str, Any]) -> None:
    """
    Save a single sensor reading (with virtual DO/NH3 fields included)
    into readings.csv for this pond.
    """
    pond_dir = ensure_pond_folder(pond_id)
    csv_path = pond_dir / "readings.csv"
    _append_csv_row(csv_path, reading)


def get_recent_readings(pond_id: str, n_steps: int) -> pd.DataFrame:
    """
    Return the last n_steps readings for this pond as a DataFrame,
    ordered by timestamp ascending.

    If file or not enough rows exist, returns as many as available (possibly 0).
    """
    pond_dir = ensure_pond_folder(pond_id)
    csv_path = pond_dir / "readings.csv"

    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    # Try to parse timestamp if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Sort by timestamp ascending
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    # Take last n_steps
    if len(df) > n_steps:
        df = df.iloc[-n_steps:]

    df = df.reset_index(drop=True)
    return df


def save_risk_prediction(pond_id: str, risk: Dict[str, Any]) -> None:
    """
    Save one risk prediction row in risk_predictions.csv.
    You can include both the inputs (temp, pH, etc.) and outputs (probs, labels).
    """
    pond_dir = ensure_pond_folder(pond_id)
    csv_path = pond_dir / "risk_predictions.csv"
    _append_csv_row(csv_path, risk)


def save_forecast(pond_id: str, forecast: Dict[str, Any]) -> None:
    """
    Save one forecast row in forecasts.csv.
    Suggested keys:
      - timestamp
      - forecast_+1h, forecast_+6h, forecast_+24h, forecast_+3d
      - risk_trend
      - model_version
    """
    pond_dir = ensure_pond_folder(pond_id)
    csv_path = pond_dir / "forecasts.csv"
    _append_csv_row(csv_path, forecast)


def save_actions(pond_id: str, actions: Dict[str, Any]) -> None:
    """
    Save a row into actions_log.csv with:
      - timestamp
      - alert_level
      - immediate_actions (str or JSON string)
      - investigation_actions
      - preventive_actions
      - optional links to forecast / risk ids
    """
    pond_dir = ensure_pond_folder(pond_id)
    csv_path = pond_dir / "actions_log.csv"
    _append_csv_row(csv_path, actions)
'''