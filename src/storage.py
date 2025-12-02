"""
storage.py

Thin wrapper that selects storage backend:
- local CSV backend (default)
- firebase backend (optional, if configured)

Fixed for Hugging Face deployment - no relative imports.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add current directory to path for imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import storage_local

BACKEND = os.getenv("POSEIDON_STORAGE_BACKEND", "local").lower()


def _backend_name() -> str:
    return BACKEND


def _get_firebase_backend():
    """
    Lazy-import firebase backend only if needed.
    """
    try:
        import storage_firebase
        return storage_firebase
    except Exception as e:
        raise RuntimeError(
            f"POSEIDON_STORAGE_BACKEND is 'firebase' but Firebase backend "
            f"could not be imported: {e}"
        )


# Public API proxy functions
def save_reading(pond_id: str, reading: Dict[str, Any]) -> None:
    if BACKEND == "local":
        return storage_local.save_reading(pond_id, reading)
    elif BACKEND == "firebase":
        fb = _get_firebase_backend()
        return fb.save_reading(pond_id, reading)
    else:
        raise ValueError(f"Unknown storage backend: {BACKEND}")


def get_recent_readings(pond_id: str, n_steps: int):
    if BACKEND == "local":
        return storage_local.get_recent_readings(pond_id, n_steps)
    elif BACKEND == "firebase":
        fb = _get_firebase_backend()
        return fb.get_recent_readings(pond_id, n_steps)
    else:
        raise ValueError(f"Unknown storage backend: {BACKEND}")


def save_risk_prediction(pond_id: str, risk: Dict[str, Any]) -> None:
    if BACKEND == "local":
        return storage_local.save_risk_prediction(pond_id, risk)
    elif BACKEND == "firebase":
        fb = _get_firebase_backend()
        return fb.save_risk_prediction(pond_id, risk)
    else:
        raise ValueError(f"Unknown storage backend: {BACKEND}")


def save_forecast(pond_id: str, forecast: Dict[str, Any]) -> None:
    if BACKEND == "local":
        return storage_local.save_forecast(pond_id, forecast)
    elif BACKEND == "firebase":
        fb = _get_firebase_backend()
        return fb.save_forecast(pond_id, forecast)
    else:
        raise ValueError(f"Unknown storage backend: {BACKEND}")


def save_actions(pond_id: str, actions: Dict[str, Any]) -> None:
    if BACKEND == "local":
        return storage_local.save_actions(pond_id, actions)
    elif BACKEND == "firebase":
        fb = _get_firebase_backend()
        return fb.save_actions(pond_id, actions)
    else:
        raise ValueError(f"Unknown storage backend: {BACKEND}")



'''
"""
storage.py

Thin wrapper that selects storage backend:
- local CSV backend (default)
- firebase backend (optional, if configured)

Usage in your code:
  from src import storage
  storage.save_reading(...)
"""

import os
from typing import Dict, Any

from . import storage_local  # always safe, purely local CSV

BACKEND = os.getenv("POSEIDON_STORAGE_BACKEND", "local").lower()


def _backend_name() -> str:
    return BACKEND


def _get_firebase_backend():
    """
    Lazy-import firebase backend so that we only touch Firestore
    if POSEIDON_STORAGE_BACKEND='firebase'.

    This avoids import-time crashes in environments where Firebase
    is not configured (e.g., Hugging Face demo).
    """
    try:
        from . import storage_firebase
    except Exception as e:
        raise RuntimeError(
            f"POSEIDON_STORAGE_BACKEND is 'firebase' but Firebase backend "
            f"could not be imported: {e}"
        )
    return storage_firebase


# -------------------------------------------------------------------
# Public API proxy functions
# -------------------------------------------------------------------

def save_reading(pond_id: str, reading: Dict[str, Any]) -> None:
    if BACKEND == "local":
        return storage_local.save_reading(pond_id, reading)
    elif BACKEND == "firebase":
        fb = _get_firebase_backend()
        return fb.save_reading(pond_id, reading)
    else:
        raise ValueError(f"Unknown storage backend: {BACKEND}")


def get_recent_readings(pond_id: str, n_steps: int):
    if BACKEND == "local":
        return storage_local.get_recent_readings(pond_id, n_steps)
    elif BACKEND == "firebase":
        fb = _get_firebase_backend()
        return fb.get_recent_readings(pond_id, n_steps)
    else:
        raise ValueError(f"Unknown storage backend: {BACKEND}")


def save_risk_prediction(pond_id: str, risk: Dict[str, Any]) -> None:
    if BACKEND == "local":
        return storage_local.save_risk_prediction(pond_id, risk)
    elif BACKEND == "firebase":
        fb = _get_firebase_backend()
        return fb.save_risk_prediction(pond_id, risk)
    else:
        raise ValueError(f"Unknown storage backend: {BACKEND}")


def save_forecast(pond_id: str, forecast: Dict[str, Any]) -> None:
    if BACKEND == "local":
        return storage_local.save_forecast(pond_id, forecast)
    elif BACKEND == "firebase":
        fb = _get_firebase_backend()
        return fb.save_forecast(pond_id, forecast)
    else:
        raise ValueError(f"Unknown storage backend: {BACKEND}")


def save_actions(pond_id: str, actions: Dict[str, Any]) -> None:
    if BACKEND == "local":
        return storage_local.save_actions(pond_id, actions)
    elif BACKEND == "firebase":
        fb = _get_firebase_backend()
        return fb.save_actions(pond_id, actions)
    else:
        raise ValueError(f"Unknown storage backend: {BACKEND}")
'''