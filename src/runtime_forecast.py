"""
Runtime hybrid forecast engine for Poseidon Project.
Fixed for src/ structure with PROPER Windows path handling.
"""

import numpy as np
import pandas as pd
from pathlib import Path, PureWindowsPath, PurePosixPath
import joblib
import tensorflow as tf
import traceback

# For files in src/, go UP to reach artifacts/
SCRIPT_DIR = Path(__file__).resolve().parent  # This is src/
ROOT = SCRIPT_DIR.parent  # This is root/ (one level up)


def extract_filename_from_any_path(path_str: str) -> str:
    """
    Extract just the filename from any path (Windows or Unix).
    
    Handles paths like:
    - C:\\Users\\PC\\...\\file.h5
    - /workspace/artifacts/file.h5
    - file.h5
    """
    # Try as Windows path first
    try:
        win_path = PureWindowsPath(path_str)
        if len(win_path.parts) > 1:  # It's actually a path
            return win_path.name
    except:
        pass
    
    # Try as Unix path
    try:
        unix_path = PurePosixPath(path_str)
        if len(unix_path.parts) > 1:  # It's actually a path
            return unix_path.name
    except:
        pass
    
    # Fallback: just take everything after the last slash or backslash
    filename = path_str.replace('\\', '/').split('/')[-1]
    return filename


def find_model_file(filename: str) -> Path:
    """Smart model file finder for src/ structure."""
    possible_paths = [
        ROOT / "artifacts" / "model_registry" / filename,
        ROOT / "artifacts" / "soft_sensors" / filename,
        ROOT / filename,
        SCRIPT_DIR / "artifacts" / "model_registry" / filename,
        SCRIPT_DIR / filename,
        Path("/workspace") / "artifacts" / "model_registry" / filename,
        Path("/workspace") / "artifacts" / "soft_sensors" / filename,
        Path("/workspace") / filename,
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Found model at: {path}")
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
    
    # VERBOSE: Show what we searched
    print(f"ERROR: Could not find {filename}")
    print(f"Searched in:")
    for path in possible_paths:
        print(f"  - {path}")
    
    raise FileNotFoundError(
        f"Model file '{filename}' not found.\n"
        f"Expected: {ROOT}/artifacts/model_registry/{filename}\n"
        f"Script in: {SCRIPT_DIR}\n"
        f"Root: {ROOT}"
    )


# Load hybrid bundle
print(f"Runtime forecast - SCRIPT_DIR: {SCRIPT_DIR}")
print(f"Runtime forecast - ROOT: {ROOT}")

try:
    BUNDLE_PATH = find_model_file("model3_forecast_hybrid_bundle.joblib")
    HYBRID = joblib.load(BUNDLE_PATH)
    print(f"✓ Loaded forecast bundle")
except Exception as e:
    print(f"✗ Failed to load forecast bundle: {e}")
    raise

FEATURE_COLS = HYBRID["feature_cols"]
SENSORS = HYBRID["sensors"]
HISTORY_STEPS = HYBRID["history_steps"]
ROLL_WINDOWS = HYBRID["roll_windows"]
USE_SLOPE = HYBRID["use_slope"]
HORIZONS = HYBRID["horizons"]
TAB_SCALER = HYBRID["scaler_tabular"]
SENS_SCALER = HYBRID["scaler_sensors"]
HYBRID_MODELS = HYBRID["hybrid_models"]

# VERBOSE: Show what models we expect
print(f"\nExpected forecast models:")
for h, meta in HYBRID_MODELS.items():
    original_path = meta['model_path']
    model_filename = extract_filename_from_any_path(original_path)
    print(f"  {h}: {model_filename} ({meta['family']}, {meta['algo']})")
    print(f"       Original path in bundle: {original_path}")
print()


def _resolve_model_path(path_str: str) -> Path:
    """
    Resolve model paths from bundle, properly handling Windows paths.
    """
    # Extract JUST the filename, handling Windows paths correctly
    filename = extract_filename_from_any_path(path_str)
    print(f"Resolving: '{path_str}' -> filename: '{filename}'")
    return find_model_file(filename)


def make_tabular_features(df, feature_cols):
    """Compute tabular engineered features."""
    out = {}
    last = df.iloc[-1]

    out["temperature"] = float(last["temperature"])
    out["pH"] = float(last["pH"])
    out["turbidity_proxy"] = float(last["turbidity_proxy"])
    out["predicted_do"] = float(last["predicted_do"])
    out["predicted_nh3"] = float(last["predicted_nh3"])

    for w in ROLL_WINDOWS:
        for col in SENSORS:
            key_mean = f"{col}_roll{w}_mean"
            key_std = f"{col}_roll{w}_std"
            key_min = f"{col}_roll{w}_min"
            key_max = f"{col}_roll{w}_max"

            window = df[col].tail(w)
            out[key_mean] = float(window.mean())
            out[key_std] = float(window.std() if len(window) > 1 else 0.0)
            out[key_min] = float(window.min())
            out[key_max] = float(window.max())

        if USE_SLOPE:
            y = df["predicted_do"].tail(w).values
            if len(y) > 1:
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0.0
            out[f"predicted_do_roll{w}_slope"] = float(slope)

    X = pd.DataFrame([out], columns=feature_cols)
    X = X.astype(float)
    X_scaled = TAB_SCALER.transform(X)
    return X_scaled


def make_sequence_tensor(df):
    """Build LSTM/GRU/CNN input tensor."""
    seq = df[SENSORS].tail(HISTORY_STEPS).astype(float).values.copy()
    seq = SENS_SCALER.transform(seq)
    seq = seq.reshape(1, HISTORY_STEPS, len(SENSORS))
    return seq


def predict_future_risk(recent_df):
    """
    Predict future risk for all horizons.
    
    Input:
        recent_df: DataFrame with columns:
            ["timestamp", "temperature", "pH", "turbidity_proxy",
             "predicted_do", "predicted_nh3"]
    
    Output:
        dict: { "+1h": prob, "+6h": prob, "+24h": prob, "+3d": prob }
    """
    assert len(recent_df) >= HISTORY_STEPS, (
        f"Need >= {HISTORY_STEPS} rows, got {len(recent_df)}"
    )
    recent_df = recent_df.sort_values("timestamp").reset_index(drop=True)

    results = {}

    for h, meta in HYBRID_MODELS.items():
        family = meta["family"]
        algo = meta["algo"]

        # VERBOSE ERROR HANDLING
        try:
            # Find the model file
            path = _resolve_model_path(meta['model_path'])
            print(f"Predicting {h}: Loading from {path}")
            
            if family == "classical":
                X = make_tabular_features(recent_df, FEATURE_COLS)
                model = joblib.load(path)
                pred = float(model.predict(X)[0])
                print(f"  ✓ {h}: {pred:.4f}")

            elif family == "deep":
                seq = make_sequence_tensor(recent_df)
                model = tf.keras.models.load_model(path, compile=False)
                pred = float(model.predict(seq, verbose=0)[0, 0])
                print(f"  ✓ {h}: {pred:.4f}")

            else:
                raise ValueError(f"Unknown family: {family}")

            results[h] = pred
            
        except FileNotFoundError as e:
            print(f"  ✗ {h}: Model file not found")
            print(f"     Error: {e}")
            results[h] = 0.0
            
        except Exception as e:
            print(f"  ✗ {h}: Prediction failed")
            print(f"     Error: {e}")
            print(f"     Traceback:")
            traceback.print_exc()
            results[h] = 0.0

    return results


if __name__ == "__main__":
    print("Hybrid forecast engine loaded.")
    print("Available horizons:", list(HYBRID_MODELS.keys()))


'''
"""
Runtime hybrid forecast engine for Poseidon Project.
Loads:
- hybrid forecast bundle (model3_forecast_hybrid_bundle.joblib)
- classical models (per horizon)
- deep models (per horizon, Keras)
Computes:
- tabular engineered features for classical horizons
- sequence (3D tensor) for deep horizons
Main API:
    predict_future_risk(recent_df)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import tensorflow as tf

# -----------------------------
# Load hybrid bundle once
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]   # /workspace or Poseidon root
MODEL_REG = ROOT / "artifacts" / "model_registry"

BUNDLE_PATH = MODEL_REG / "model3_forecast_hybrid_bundle.joblib"
HYBRID = joblib.load(BUNDLE_PATH)

FEATURE_COLS   = HYBRID["feature_cols"]
SENSORS        = HYBRID["sensors"]
HISTORY_STEPS  = HYBRID["history_steps"]
ROLL_WINDOWS   = HYBRID["roll_windows"]
USE_SLOPE      = HYBRID["use_slope"]
HORIZONS       = HYBRID["horizons"]
TAB_SCALER     = HYBRID["scaler_tabular"]
SENS_SCALER    = HYBRID["scaler_sensors"]
HYBRID_MODELS  = HYBRID["hybrid_models"]  # meta per horizon


# ---------------------------------------------------------
# Helper: resolve model paths from bundle
# ---------------------------------------------------------
def _resolve_model_path(path_str: str) -> Path:
    """
    Make sure model paths from the hybrid bundle work inside the container.

    The bundle may contain absolute Windows paths like:
      C:\\Users\\PC\\...\\model3_deep_+1h_lstm.h5

    Strategy:
      1) Take only the filename part.
      2) Look for it under artifacts/model_registry.
      3) If not found, search the whole project tree.
      4) If still not found, raise a clear error.
    """
    p = Path(path_str)
    filename = p.name  # ALWAYS ignore the Windows directories

    # 1) Try artifacts/model_registry/<filename>
    candidate = MODEL_REG / filename
    if candidate.exists():
        return candidate

    # 2) Search entire project for that filename
    matches = list(ROOT.rglob(filename))
    if matches:
        return matches[0]

    # 3) Fail loud with a clear message
    raise FileNotFoundError(
        f"Could not resolve model path from bundle: {path_str}. "
        f"Looked for '{filename}' under {MODEL_REG} and project root {ROOT}."
    )


# ---------------------------------------------------------
# Helper: classical feature engineering (same as notebook)
# ---------------------------------------------------------
def make_tabular_features(df, feature_cols):
    """Given a recent-data DataFrame (multi-row), compute tabular engineered features
    for classical models. Mirrors notebook logic but simplified for inference."""
    out = {}
    last = df.iloc[-1]

    # basic sensors
    out["temperature"]     = float(last["temperature"])
    out["pH"]              = float(last["pH"])
    out["turbidity_proxy"] = float(last["turbidity_proxy"])
    out["predicted_do"]    = float(last["predicted_do"])
    out["predicted_nh3"]   = float(last["predicted_nh3"])

    # rolling windows
    for w in ROLL_WINDOWS:
        for col in SENSORS:
            key_mean = f"{col}_roll{w}_mean"
            key_std  = f"{col}_roll{w}_std"
            key_min  = f"{col}_roll{w}_min"
            key_max  = f"{col}_roll{w}_max"

            window = df[col].tail(w)
            out[key_mean] = float(window.mean())
            out[key_std]  = float(window.std() if len(window) > 1 else 0.0)
            out[key_min]  = float(window.min())
            out[key_max]  = float(window.max())

        if USE_SLOPE:
            # slope on predicted_do
            y = df["predicted_do"].tail(w).values
            if len(y) > 1:
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0.0
            out[f"predicted_do_roll{w}_slope"] = float(slope)

    # Convert to DataFrame
    X = pd.DataFrame([out], columns=feature_cols)
    X = X.astype(float)

    # scale
    X_scaled = TAB_SCALER.transform(X)
    return X_scaled  # numpy array shape (1, n_features)


# -----------------------------------------------------
# Helper: build LSTM/GRU/CNN input (same as training)
# -----------------------------------------------------
def make_sequence_tensor(df):
    """
    Given a recent-data DataFrame (>= HISTORY_STEPS rows),
    build a (1, HISTORY_STEPS, num_sensors) tensor for deep models.
    """
    seq = df[SENSORS].tail(HISTORY_STEPS).astype(float).values.copy()
    # Scale sensors
    seq = SENS_SCALER.transform(seq)
    # reshape for keras
    seq = seq.reshape(1, HISTORY_STEPS, len(SENSORS))
    return seq


# -----------------------------------------------------
# Main function: hybrid prediction per horizon
# -----------------------------------------------------
def predict_future_risk(recent_df):
    """
    Input:
        recent_df: DataFrame with columns:
            ["timestamp", "temperature", "pH", "turbidity_proxy",
             "predicted_do", "predicted_nh3"]
            Must contain at least HISTORY_STEPS rows.

    Output:
        dict:
            { "+1h": prob_high, "+6h": prob_high, "+24h": prob_high, "+3d": prob_high }
    """
    assert len(recent_df) >= HISTORY_STEPS, (
        f"Need >= {HISTORY_STEPS} rows, got {len(recent_df)}"
    )
    recent_df = recent_df.sort_values("timestamp").reset_index(drop=True)

    results = {}

    for h, meta in HYBRID_MODELS.items():
        family = meta["family"]      # "classical" or "deep"
        algo   = meta["algo"]        # e.g. "extra" or "lstm"

        # --- CRITICAL FIX: ignore any absolute path stored in the bundle ---
        
        raw_model_path = meta["model_path"]
        model_name = Path(str(raw_model_path)).name  # e.g. "model3_deep_+1h_lstm.h5"
        path = MODEL_REG / model_name               # /workspace/artifacts/model_registry/<name>

        if not path.exists():
            raise FileNotFoundError(
                f"Model file for horizon {h} not found at {path}. "
                f"(raw path in bundle was: {raw_model_path})"
            )
        
        path = _resolve_model_path(meta["model_path"])
        if family == "classical":
            # Classical tabular pipeline
            X = make_tabular_features(recent_df, FEATURE_COLS)
            model = joblib.load(path)
            pred = float(model.predict(X)[0])

        elif family == "deep":
            # Deep model pipeline (LSTM/GRU/CNN)
            seq = make_sequence_tensor(recent_df)
            model = tf.keras.models.load_model(path, compile=False)
            pred = float(model.predict(seq)[0, 0])

        else:
            raise ValueError(f"Unknown family: {family}")

        results[h] = pred

    return results


# -------------------------
# If run as script: quick test
# -------------------------
if __name__ == "__main__":
    print("Hybrid forecast engine loaded.")
    print("ROOT:", ROOT)
    print("MODEL_REG:", MODEL_REG)
    print("Available horizons:", list(HYBRID_MODELS.keys()))

'''
