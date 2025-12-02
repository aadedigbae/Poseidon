"""
storage_firebase.py

Firestore-based storage backend for Poseidon.

Schema:

  ponds/{pond_id}/sensor_readings/{auto_id}
  ponds/{pond_id}/risk_predictions/{auto_id}
  ponds/{pond_id}/forecasts/{auto_id}
  ponds/{pond_id}/actions_log/{auto_id}
"""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import os

import pandas as pd
from google.cloud import firestore
from google.oauth2 import service_account


# -------------------------------------------------------------------
# Firestore Client Initialization (lazy)
# -------------------------------------------------------------------

_db = None  # will be created on first use


def _get_firestore_client() -> firestore.Client:
    """
    Initialize Firestore client using a service account JSON pointed to by:
      POSEIDON_FIREBASE_CREDENTIALS

    This is called lazily, NOT at import time.
    """
    global _db
    if _db is not None:
        return _db

    cred_path = os.getenv("POSEIDON_FIREBASE_CREDENTIALS")
    if not cred_path:
        raise RuntimeError(
            "POSEIDON_FIREBASE_CREDENTIALS is not set, but Firebase storage "
            "is enabled. Set this env var to point to your service account file."
        )

    cred_path = Path(cred_path).expanduser().resolve()
    if not cred_path.exists():
        raise FileNotFoundError(
            f"Firebase service account JSON not found at: {cred_path}"
        )

    creds = service_account.Credentials.from_service_account_file(str(cred_path))
    _db = firestore.Client(credentials=creds, project=creds.project_id)
    return _db


'''
def _get_firestore_client() -> firestore.Client:
    """
    Initialize Firestore client using a service account JSON pointed to by:
      POSEIDON_FIREBASE_CREDENTIALS
    """
    cred_path = os.getenv("POSEIDON_FIREBASE_CREDENTIALS")
    if not cred_path:
        raise RuntimeError(
            "Environment variable POSEIDON_FIREBASE_CREDENTIALS is not set. "
            "It should point to your Firebase service account JSON file."
        )

    cred_path = Path(cred_path).expanduser().resolve()
    if not cred_path.exists():
        raise FileNotFoundError(
            f"Firebase service account JSON not found at: {cred_path}"
        )

    creds = service_account.Credentials.from_service_account_file(str(cred_path))
    client = firestore.Client(credentials=creds, project=creds.project_id)
    return client


_db = _get_firestore_client()
'''

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _normalize_for_firestore(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Firestore can store timestamps as datetime objects.
    We make sure Python datetime is used where appropriate.
    """
    clean = {}
    for k, v in data.items():
        if isinstance(v, str):
            # Try to parse ISO datetime strings into datetime objects, but don't crash
            if "T" in v or (" " in v and ":" in v and "-" in v):
                try:
                    dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
                    clean[k] = dt
                    continue
                except Exception:
                    pass
        clean[k] = v
    return clean

def _get_pond_collection(pond_id: str, subcollection: str):
    """
    Helper to get a reference to a pond's subcollection.
    """
    client = _get_firestore_client()
    return client.collection("ponds").document(pond_id).collection(subcollection)

'''
def _get_pond_collection(pond_id: str, subcollection: str):
    """
    Helper to get a reference to a pond's subcollection.
    """
    return _db.collection("ponds").document(pond_id).collection(subcollection)
'''

# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def save_reading(pond_id: str, reading: Dict[str, Any]) -> None:
    """
    Save a single sensor reading in:
      ponds/{pond_id}/sensor_readings/{auto_id}
    """
    clean = _normalize_for_firestore(reading)
    coll = _get_pond_collection(pond_id, "sensor_readings")
    coll.add(clean)  # auto document ID


def get_recent_readings(pond_id: str, n_steps: int) -> pd.DataFrame:
    """
    Return the last n_steps readings for this pond as a DataFrame,
    ordered by timestamp ascending.
    """
    coll = _get_pond_collection(pond_id, "sensor_readings")
    # We assume there is a "timestamp" field stored as Firestore Timestamp
    q = coll.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(n_steps)

    docs = list(q.stream())
    rows = [d.to_dict() for d in docs]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def save_risk_prediction(pond_id: str, risk: Dict[str, Any]) -> None:
    """
    Save a risk prediction document in:
      ponds/{pond_id}/risk_predictions/{auto_id}
    """
    clean = _normalize_for_firestore(risk)
    coll = _get_pond_collection(pond_id, "risk_predictions")
    coll.add(clean)


def save_forecast(pond_id: str, forecast: Dict[str, Any]) -> None:
    """
    Save a forecast snapshot in:
      ponds/{pond_id}/forecasts/{auto_id}
    Suggested keys:
      - timestamp
      - forecast_+1h, forecast_+6h, forecast_+24h, forecast_+3d
      - risk_trend
      - model_version
    """
    clean = _normalize_for_firestore(forecast)
    coll = _get_pond_collection(pond_id, "forecasts")
    coll.add(clean)


def save_actions(pond_id: str, actions: Dict[str, Any]) -> None:
    """
    Save an actions / recommendations document in:
      ponds/{pond_id}/actions_log/{auto_id}
    """
    clean = _normalize_for_firestore(actions)
    coll = _get_pond_collection(pond_id, "actions_log")
    coll.add(clean)
