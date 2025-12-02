"""
soft_sensors_runtime.py

Runtime helpers for virtual sensors.
Fixed for src/ directory structure in Hugging Face.
"""

from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np

# For files in src/, go UP to reach artifacts/
SCRIPT_DIR = Path(__file__).resolve().parent  # This is src/
ROOT = SCRIPT_DIR.parent  # This is root/ (one level up)


def find_model_file(filename: str) -> Path:
    """Smart model file finder for src/ structure."""
    possible_paths = [
        # Most likely (your structure)
        ROOT / "artifacts" / "soft_sensors" / filename,
        ROOT / "artifacts" / "model_registry" / filename,
        # Alternatives
        ROOT / filename,
        SCRIPT_DIR / "artifacts" / "soft_sensors" / filename,
        SCRIPT_DIR / filename,
        # Container paths
        Path("/workspace") / "artifacts" / "soft_sensors" / filename,
        Path("/workspace") / "artifacts" / "model_registry" / filename,
        Path("/workspace") / filename,
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Found sensor model at: {path}")
            return path
    
    # Recursive search
    try:
        for search_root in [ROOT, SCRIPT_DIR, Path("/workspace")]:
            matches = list(search_root.rglob(filename))
            if matches:
                print(f"Found via search: {matches[0]}")
                return matches[0]
    except:
        pass
    
    raise FileNotFoundError(
        f"Sensor model '{filename}' not found.\n"
        f"Expected: {ROOT}/artifacts/soft_sensors/{filename}\n"
        f"Script in: {SCRIPT_DIR}\n"
        f"Root: {ROOT}"
    )


# Initialize
print(f"Soft sensors - SCRIPT_DIR: {SCRIPT_DIR}")
print(f"Soft sensors - ROOT: {ROOT}")

try:
    VM_DO_PATH = find_model_file("virtual_do.joblib")
    VM_NH3_PATH = find_model_file("virtual_nh3.joblib")
    
    vm_do_bundle = joblib.load(VM_DO_PATH)
    vm_nh3_bundle = joblib.load(VM_NH3_PATH)
    
    print(f"✓ Loaded DO sensor")
    print(f"✓ Loaded NH3 sensor")
    
except Exception as e:
    print(f"✗ Failed to load sensor models: {e}")
    raise

vm_do_model = vm_do_bundle["model"]
vm_do_features = vm_do_bundle["features"]

vm_nh3_model = vm_nh3_bundle["model"]
vm_nh3_features = vm_nh3_bundle["features"]


def enrich_with_virtual_sensors(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add predicted_do and predicted_nh3 to a raw reading.
    
    Input:
        raw: dict with at least "temperature", "pH", "turbidity_proxy"
    
    Output:
        dict: copy of raw with added "predicted_do" and "predicted_nh3"
    """
    enriched = dict(raw)

    # DO
    do_vec = []
    for fname in vm_do_features:
        do_vec.append(float(raw.get(fname, 0.0)))
    do_arr = np.array(do_vec, dtype=float).reshape(1, -1)
    pred_do = float(vm_do_model.predict(do_arr)[0])

    # NH3
    nh3_vec = []
    for fname in vm_nh3_features:
        nh3_vec.append(float(raw.get(fname, 0.0)))
    nh3_arr = np.array(nh3_vec, dtype=float).reshape(1, -1)
    pred_nh3 = float(vm_nh3_model.predict(nh3_arr)[0])

    enriched["predicted_do"] = pred_do
    enriched["predicted_nh3"] = pred_nh3

    return enriched


if __name__ == "__main__":
    print("Soft sensors loaded successfully.")
    print("DO features:", vm_do_features)
    print("NH3 features:", vm_nh3_features)


''''
"""
Runtime wrapper around virtual soft sensors for DO and NH3.

Uses:
  artifacts/soft_sensors/virtual_do.joblib
  artifacts/soft_sensors/virtual_nh3.joblib

Each joblib file is expected to store a dict:
  {
    "model": fitted_regressor,
    "features": ["temperature", "pH", "turbidity_proxy", ...]
  }
"""



from pathlib import Path
import numpy as np
import pandas as pd
import joblib


# Locate project root
def find_project_root(project_name: str = "Poseidon") -> Path:
    cwd = Path.cwd().resolve()
    pl = project_name.lower()
    for p in [cwd] + list(cwd.parents):
        if p.name.lower() == pl:
            return p
    if cwd.name.lower() == "notebooks" and cwd.parent.exists():
        return cwd.parent
    raise FileNotFoundError(f"Could not locate project root '{project_name}'. Starting cwd: {cwd}")


ROOT = find_project_root("Poseidon")
ART  = ROOT / "artifacts"
SOFT = ART / "soft_sensors"

do_path  = SOFT / "virtual_do.joblib"
nh3_path = SOFT / "virtual_nh3.joblib"

vm_do  = joblib.load(do_path)
vm_nh3 = joblib.load(nh3_path)

do_model  = vm_do["model"]
do_feats  = vm_do["features"]
nh3_model = vm_nh3["model"]
nh3_feats = vm_nh3["features"]


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for soft sensors: {missing}")
    return df


def estimate_virtual_do(df: pd.DataFrame) -> np.ndarray:
    """
    Compute virtual DO for each row in df using vm_do.
    Returns a numpy array; also safe to assign as df['predicted_do'].
    """
    _ensure_columns(df, do_feats)
    X = df[do_feats].astype(float).values
    return do_model.predict(X)


def estimate_virtual_nh3(df: pd.DataFrame) -> np.ndarray:
    """
    Compute virtual NH3 for each row in df using vm_nh3.
    """
    _ensure_columns(df, nh3_feats)
    X = df[nh3_feats].astype(float).values
    return nh3_model.predict(X)


def run_soft_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich df with predicted_do and predicted_nh3 columns if missing.
    Returns a new DataFrame (does not modify in-place).
    """
    df = df.copy()

    if "predicted_do" not in df.columns:
        df["predicted_do"] = estimate_virtual_do(df)

    if "predicted_nh3" not in df.columns:
        df["predicted_nh3"] = estimate_virtual_nh3(df)

    return df


if __name__ == "__main__":
    print("Soft sensors runtime loaded.")
    print("DO features:", do_feats)
    print("NH3 features:", nh3_feats)
'''


#---------------------------------------------------
#New page
#---------------------------------------------------

"""
soft_sensors_runtime.py

Runtime helpers for virtual sensors:
- Virtual DO estimator
- Virtual NH3 estimator

We assume the models were trained in notebook 01 and saved as:
  artifacts/soft_sensors/virtual_do.joblib
  artifacts/soft_sensors/virtual_nh3.joblib

Each joblib file holds a dict like:
{
  "model": fitted_regressor,
  "features": ["temperature","pH","turbidity_proxy", ...],
  "version": "v1",
  "notes": "..."
}
"""
'''
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np

# -------------------------------------------------------------------
# Locate project root and model paths
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]   # Poseidon/
ARTIFACTS = ROOT / "artifacts"
SOFT_DIR = ARTIFACTS / "soft_sensors"

VM_DO_PATH = SOFT_DIR / "virtual_do.joblib"
VM_NH3_PATH = SOFT_DIR / "virtual_nh3.joblib"

# -------------------------------------------------------------------
# Load bundles (fail loud if missing)
# -------------------------------------------------------------------

if not VM_DO_PATH.exists() or not VM_NH3_PATH.exists():
    raise FileNotFoundError(
        f"Virtual sensor bundles not found. Expected:\n"
        f"  {VM_DO_PATH}\n"
        f"  {VM_NH3_PATH}\n"
        f"Make sure you ran 01_soft_sensors_DO_NH3 and saved the models."
    )

vm_do_bundle = joblib.load(VM_DO_PATH)
vm_nh3_bundle = joblib.load(VM_NH3_PATH)

vm_do_model = vm_do_bundle["model"]
vm_do_features = vm_do_bundle["features"]

vm_nh3_model = vm_nh3_bundle["model"]
vm_nh3_features = vm_nh3_bundle["features"]


# -------------------------------------------------------------------
# Public helper
# -------------------------------------------------------------------

def enrich_with_virtual_sensors(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take a raw reading dict that has (at least):
      - "temperature"
      - "pH"
      - "turbidity_proxy"
    and return a NEW dict that includes:
      - "predicted_do"
      - "predicted_nh3"

    We assume the feature names in the bundles match keys in `raw`.
    Missing features are treated as 0.0 (you can adjust if needed).
    """
    enriched = dict(raw)  # copy

    # Build feature vector for DO
    do_vec = []
    for fname in vm_do_features:
        do_vec.append(float(raw.get(fname, 0.0)))
    do_arr = np.array(do_vec, dtype=float).reshape(1, -1)
    pred_do = float(vm_do_model.predict(do_arr)[0])

    # Build feature vector for NH3
    nh3_vec = []
    for fname in vm_nh3_features:
        nh3_vec.append(float(raw.get(fname, 0.0)))
    nh3_arr = np.array(nh3_vec, dtype=float).reshape(1, -1)
    pred_nh3 = float(vm_nh3_model.predict(nh3_arr)[0])

    enriched["predicted_do"] = pred_do
    enriched["predicted_nh3"] = pred_nh3

    return enriched
'''
