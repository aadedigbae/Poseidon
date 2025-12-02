"""
Decision support layer for Poseidon.
Fixed for src/ directory structure in Hugging Face.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

# For files in src/, we need to go UP to reach artifacts/
SCRIPT_DIR = Path(__file__).resolve().parent  # This is src/
ROOT = SCRIPT_DIR.parent  # This is root/ (one level up)

# Add src to path for imports
sys.path.insert(0, str(SCRIPT_DIR))

from runtime_forecast import predict_future_risk


def find_model_file(filename: str) -> Path:
    """
    Smart model file finder for src/ structure.
    """
    possible_paths = [
        # Most likely location (for your structure)
        ROOT / "artifacts" / "model_registry" / filename,
        # Alternative locations
        ROOT / "artifacts" / "soft_sensors" / filename,
        ROOT / filename,
        SCRIPT_DIR / "artifacts" / "model_registry" / filename,
        SCRIPT_DIR / filename,
        # Container paths
        Path("/workspace") / "artifacts" / "model_registry" / filename,
        Path("/workspace") / "artifacts" / "soft_sensors" / filename,
        Path("/workspace") / filename,
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Found model at: {path}")
            return path
    
    # Recursive search as last resort
    try:
        for search_root in [ROOT, SCRIPT_DIR, Path("/workspace")]:
            matches = list(search_root.rglob(filename))
            if matches:
                print(f"Found via search: {matches[0]}")
                return matches[0]
    except:
        pass
    
    # Debug info
    print(f"ERROR: Could not find {filename}")
    print(f"SCRIPT_DIR (src/): {SCRIPT_DIR}")
    print(f"ROOT (parent): {ROOT}")
    print(f"Searched in:")
    for path in possible_paths:
        print(f"  - {path} (exists: {path.exists()})")
    
    print(f"\nFiles in ROOT:")
    try:
        for item in ROOT.iterdir():
            print(f"  - {item.name}")
    except:
        pass
    
    print(f"\nFiles in artifacts/ (if exists):")
    artifacts = ROOT / "artifacts"
    if artifacts.exists():
        for item in artifacts.rglob("*"):
            if item.is_file():
                print(f"  - {item.relative_to(ROOT)}")
    
    raise FileNotFoundError(
        f"Model file '{filename}' not found.\n"
        f"Expected location: {ROOT}/artifacts/model_registry/{filename}\n"
        f"Script is in: {SCRIPT_DIR}\n"
        f"Root is: {ROOT}"
    )


# Load models
print("=" * 60)
print("LOADING MODELS")
print("=" * 60)
print(f"Script location: {SCRIPT_DIR}")
print(f"Root directory: {ROOT}")
print(f"Current working directory: {Path.cwd()}")

try:
    hybrid_path = find_model_file("model3_forecast_hybrid_bundle.joblib")
    hybrid = joblib.load(hybrid_path)
    print(f"✓ Loaded hybrid forecast bundle")
except Exception as e:
    print(f"✗ Failed to load hybrid forecast bundle: {e}")
    raise

try:
    model1_path = find_model_file("model1_risk_classifier.joblib")
    risk_bundle = joblib.load(model1_path)
    print(f"✓ Loaded risk classifier")
except Exception as e:
    print(f"✗ Failed to load risk classifier: {e}")
    raise

# Extract configuration
feature_cols = hybrid["feature_cols"]
SENSORS = hybrid["sensors"]
HISTORY_STEPS = hybrid["history_steps"]
ROLL_WINDOWS = hybrid["roll_windows"]
USE_SLOPE = hybrid["use_slope"]
HORIZONS = hybrid["horizons"]
TAB_SCALER = hybrid["scaler_tabular"]
SENS_SCALER = hybrid["scaler_sensors"]
HYBRID_MODELS = hybrid["hybrid_models"]
risk_col = hybrid["risk_label_column"]

risk_model = risk_bundle["model"]
risk_features = risk_bundle["features"]
bundle_classes = risk_bundle["classes"]
risk_thresholds = risk_bundle.get("thresholds", {})
risk_version = risk_bundle.get("version", "unknown")
risk_notes = risk_bundle.get("notes", "")

model_class_order = risk_model.classes_

print(f"✓ Configuration loaded: {len(SENSORS)} sensors, {HISTORY_STEPS} history steps")
print("=" * 60)


def get_risk_label_from_proba(proba_vec: np.ndarray) -> str:
    idx = int(np.argmax(proba_vec))
    return model_class_order[idx]


def prepare_risk_features_from_row(row: pd.Series) -> np.ndarray:
    vals = []
    for f in risk_features:
        if f in row.index:
            vals.append(row[f])
        else:
            vals.append(0.0)
    X = np.array(vals, dtype=float).reshape(1, -1)
    return X


PARAM_THRESHOLDS = {
    "predicted_do": {
        "low_critical": 3.0,
        "low_warning": 4.0,
        "optimal_min": 4.5,
        "optimal_max": 7.5,
    },
    "predicted_nh3": {
        "high_critical": 0.5,
        "high_warning": 0.25,
    },
    "temperature": {
        "low": 24.0,
        "high": 32.0,
    },
    "pH": {
        "low": 6.5,
        "high": 9.0,
    },
    "turbidity_proxy": {
        "high_warning": 0.7,
    },
}


def compute_trends(recent_df: pd.DataFrame, horizon_hours: float = 6.0) -> dict:
    recent_df = recent_df.sort_values("timestamp").reset_index(drop=True)
    if len(recent_df) < 2:
        return {}

    t_end = recent_df["timestamp"].max()
    window_start = t_end - pd.Timedelta(hours=horizon_hours)
    window_df = recent_df[recent_df["timestamp"] >= window_start]
    if len(window_df) < 2:
        window_df = recent_df.tail(2)

    trends = {}

    for col in ["temperature", "pH", "turbidity_proxy", "predicted_do", "predicted_nh3"]:
        arr = window_df[col].values.astype(float)
        start = arr[0]
        end = arr[-1]
        delta = end - start

        x = np.arange(len(arr))
        slope = np.polyfit(x, arr, 1)[0] if len(arr) > 1 else 0.0

        trends[col] = {
            "start": float(start),
            "end": float(end),
            "delta": float(delta),
            "slope": float(slope),
            "n_points": int(len(arr)),
        }

    return trends


def explain_current_risk(recent_df: pd.DataFrame, horizon_hours: float = 6.0) -> dict:
    recent_df = recent_df.sort_values("timestamp").reset_index(drop=True)
    last = recent_df.iloc[-1]

    X_risk = prepare_risk_features_from_row(last)
    proba = risk_model.predict_proba(X_risk)[0]
    label_argmax = get_risk_label_from_proba(proba)

    if "HIGH" in model_class_order:
        idx_high = model_class_order.tolist().index("HIGH")
        p_high = float(proba[idx_high])
    else:
        p_high = 0.0

    high_thr = risk_thresholds.get("HIGH", 0.3)
    final_label = "HIGH" if p_high >= high_thr else label_argmax

    trends = compute_trends(recent_df, horizon_hours=horizon_hours)
    factors = []

    do = float(last["predicted_do"])
    do_thr = PARAM_THRESHOLDS["predicted_do"]
    if do < do_thr["low_critical"]:
        factors.append(("PRIMARY", f"Dissolved oxygen critically low ({do:.2f} mg/L)."))
    elif do < do_thr["low_warning"]:
        factors.append(("PRIMARY", f"Dissolved oxygen in risky zone ({do:.2f} mg/L)."))

    nh3 = float(last["predicted_nh3"])
    nh3_thr = PARAM_THRESHOLDS["predicted_nh3"]
    if nh3 > nh3_thr["high_critical"]:
        factors.append(("PRIMARY", f"Ammonia critically high ({nh3:.3f} mg/L)."))
    elif nh3 > nh3_thr["high_warning"]:
        factors.append(("CONTRIBUTING", f"Ammonia elevated ({nh3:.3f} mg/L)."))

    temp = float(last["temperature"])
    t_thr = PARAM_THRESHOLDS["temperature"]
    if temp > t_thr["high"]:
        factors.append(("CONTRIBUTING", f"Water temperature high ({temp:.1f} °C)."))
    elif temp < t_thr["low"]:
        factors.append(("CONTRIBUTING", f"Water temperature low ({temp:.1f} °C)."))

    ph = float(last["pH"])
    ph_thr = PARAM_THRESHOLDS["pH"]
    if ph < ph_thr["low"] or ph > ph_thr["high"]:
        factors.append(("CONTRIBUTING", f"pH out of comfort range ({ph:.2f})."))

    turb = float(last["turbidity_proxy"])
    turb_thr = PARAM_THRESHOLDS["turbidity_proxy"]["high_warning"]
    if turb > turb_thr:
        factors.append(("SECONDARY", f"Turbidity elevated (proxy={turb:.3f})."))

    for col, info in trends.items():
        delta = info["delta"]
        npts = info["n_points"]
        if col == "predicted_do" and delta < -0.5:
            factors.append(("PRIMARY", f"Dissolved oxygen dropped {delta:.2f} mg/L over last {npts} readings."))
        if col == "predicted_nh3" and delta > 0.1:
            factors.append(("CONTRIBUTING", f"Ammonia rising by {delta:.3f} mg/L in recent window."))

    if not factors:
        factors.append((
            "INFO",
            "No strong stressors detected based on thresholds; "
            "risk level is driven by the model's learned patterns from historical data."
        ))

    explanation = {
        "risk_label": final_label,
        "risk_label_argmax": label_argmax,
        "risk_proba": {cls: float(p) for cls, p in zip(model_class_order, proba)},
        "p_high": p_high,
        "high_threshold": high_thr,
        "factors": factors,
    }

    return explanation


RECOMMENDATION_RULES = {
    "PRIMARY_DO_LOW": {
        "trigger": lambda ctx: ctx["last"]["predicted_do"] < PARAM_THRESHOLDS["predicted_do"]["low_warning"],
        "immediate": [
            "Increase aeration immediately.",
            "Avoid feeding until dissolved oxygen stabilizes.",
        ],
        "investigation": [
            "Check aerator performance and power supply.",
            "Observe fish behavior near the surface.",
        ],
        "preventive": [
            "Schedule DO measurements at dawn and late afternoon daily.",
            "Optimize feeding schedule to reduce excess organic load.",
        ],
    },
    "PRIMARY_NH3_HIGH": {
        "trigger": lambda ctx: ctx["last"]["predicted_nh3"] > PARAM_THRESHOLDS["predicted_nh3"]["high_warning"],
        "immediate": [
            "Reduce feeding by 50% for the next 24 hours.",
            "Increase partial water exchange if possible.",
        ],
        "investigation": [
            "Check for uneaten feed and sludge accumulation.",
            "Review recent changes in feeding or stocking density.",
        ],
        "preventive": [
            "Implement regular sludge removal.",
            "Align stocking densities with pond capacity.",
        ],
    },
    "TEMP_OUT_OF_RANGE": {
        "trigger": lambda ctx: (
            ctx["last"]["temperature"] > PARAM_THRESHOLDS["temperature"]["high"]
            or ctx["last"]["temperature"] < PARAM_THRESHOLDS["temperature"]["low"]
        ),
        "immediate": [
            "Avoid major feeding during hottest hours of the day.",
        ],
        "investigation": [
            "Check shading, water depth, and inflow temperature.",
        ],
        "preventive": [
            "Plan partial water exchange during cooler hours.",
            "Increase shading structures if persistent high temperatures.",
        ],
    },
}


def _dedup(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def generate_recommendations(recent_df: pd.DataFrame, explanation: dict) -> dict:
    recent_df = recent_df.sort_values("timestamp").reset_index(drop=True)
    last = recent_df.iloc[-1].to_dict()
    ctx = {"last": last, "explanation": explanation}

    immediate = []
    investigation = []
    preventive = []

    for name, rule in RECOMMENDATION_RULES.items():
        try:
            if rule["trigger"](ctx):
                immediate.extend(rule.get("immediate", []))
                investigation.extend(rule.get("investigation", []))
                preventive.extend(rule.get("preventive", []))
        except Exception as e:
            print(f"Rule {name} error: {e}")

    risk_label = explanation.get("risk_label", "LOW")

    if not immediate and not investigation and not preventive:
        if risk_label == "LOW":
            preventive = [
                "Maintain current aeration and feeding schedule.",
                "Log sensor readings at least twice per day.",
            ]
        elif risk_label == "MEDIUM":
            immediate = [
                "Increase observation frequency for the next 24 hours.",
            ]
            investigation = [
                "Check for any changes in feeding, stocking, or weather.",
            ]
            preventive = [
                "Plan a routine water quality check (DO, NH3) within the next 48 hours.",
            ]
        elif risk_label == "HIGH":
            immediate = [
                "Increase aeration immediately and avoid feeding until conditions stabilize.",
            ]
            investigation = [
                "Inspect fish behavior (gasping at surface) and verify aerator functioning.",
            ]
            preventive = [
                "Review stocking density and feeding rates to reduce long-term stress.",
            ]

    return {
        "immediate": _dedup(immediate),
        "investigation": _dedup(investigation),
        "preventive": _dedup(preventive),
    }


def build_decision_support_payload(recent_df: pd.DataFrame) -> dict:
    recent_df = recent_df.sort_values("timestamp").reset_index(drop=True)
    assert len(recent_df) >= HISTORY_STEPS, f"Need >= {HISTORY_STEPS} rows, got {len(recent_df)}"

    explanation = explain_current_risk(recent_df, horizon_hours=6.0)
    forecast = predict_future_risk(recent_df)
    recommendations = generate_recommendations(recent_df, explanation)

    max_forecast = max(forecast.values())
    current_label = explanation["risk_label"]

    if current_label == "HIGH" or max_forecast > 0.7:
        alert_level = "CRITICAL"
    elif max_forecast > 0.4:
        alert_level = "WARNING"
    else:
        alert_level = "INFO"

    payload = {
        "current_risk": explanation,
        "forecast": forecast,
        "recommendations": recommendations,
        "alert_level": alert_level,
        "latest_timestamp": str(recent_df["timestamp"].max()),
    }
    return payload


if __name__ == "__main__":
    print("Decision support module loaded successfully.")


'''
"""
Decision support layer for Poseidon:
- Explain current risk (Model 1 risk classifier)
- Generate recommendations
- Compute alert level
- Use hybrid forecaster for future risk

Public entrypoint:
    build_decision_support_payload(recent_df: pd.DataFrame) -> dict
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from .runtime_forecast import predict_future_risk


# --------------------------
# Locate project root
# --------------------------

#def find_project_root(project_name: str = "Poseidon") -> Path:
#    cwd = Path.cwd().resolve()
    pl = project_name.lower()
    for p in [cwd] + list(cwd.parents):
        if p.name.lower() == pl:
            return p
    if cwd.name.lower() == "notebooks" and cwd.parent.exists():
        return cwd.parent
    raise FileNotFoundError(f"Could not locate project root '{project_name}'. Starting cwd: {cwd}")
'''
'''
ROOT      = Path(__file__).resolve().parents[1] 
DATA      = ROOT / "data"
INTERIM   = DATA / "interim"
ART       = ROOT / "artifacts"
MODEL_REG = ART / "model_registry"

# --------------------------
# Load hybrid forecast bundle
# --------------------------
hybrid_path = MODEL_REG / "model3_forecast_hybrid_bundle.joblib"
hybrid = joblib.load(hybrid_path)

feature_cols   = hybrid["feature_cols"]
SENSORS        = hybrid["sensors"]
HISTORY_STEPS  = hybrid["history_steps"]
ROLL_WINDOWS   = hybrid["roll_windows"]
USE_SLOPE      = hybrid["use_slope"]
HORIZONS       = hybrid["horizons"]
TAB_SCALER     = hybrid["scaler_tabular"]
SENS_SCALER    = hybrid["scaler_sensors"]
HYBRID_MODELS  = hybrid["hybrid_models"]
risk_col       = hybrid["risk_label_column"]

# --------------------------
# Load Model 1 risk classifier
# --------------------------
model1_path = MODEL_REG / "model1_risk_classifier.joblib"
risk_bundle = joblib.load(model1_path)

risk_model      = risk_bundle["model"]         # CalibratedClassifierCV
risk_features   = risk_bundle["features"]      # list
bundle_classes  = risk_bundle["classes"]       # ['HIGH','LOW','MEDIUM']
risk_thresholds = risk_bundle.get("thresholds", {})
risk_version    = risk_bundle.get("version", "unknown")
risk_notes      = risk_bundle.get("notes", "")

# internal class order for predict_proba
model_class_order = risk_model.classes_


def get_risk_label_from_proba(proba_vec: np.ndarray) -> str:
    idx = int(np.argmax(proba_vec))
    return model_class_order[idx]


def prepare_risk_features_from_row(row: pd.Series) -> np.ndarray:
    """
    Align row with the training features for Model 1.
    """
    vals = []
    for f in risk_features:
        if f in row.index:
            vals.append(row[f])
        else:
            vals.append(0.0)
    X = np.array(vals, dtype=float).reshape(1, -1)
    return X


# -----------------------------
# Thresholds for explanation
# -----------------------------
PARAM_THRESHOLDS = {
    "predicted_do": {
        "low_critical": 3.0,
        "low_warning":  4.0,
        "optimal_min":  4.5,
        "optimal_max":  7.5,
    },
    "predicted_nh3": {
        "high_critical": 0.5,
        "high_warning":  0.25,
    },
    "temperature": {
        "low": 24.0,
        "high": 32.0,
    },
    "pH": {
        "low": 6.5,
        "high": 9.0,
    },
    "turbidity_proxy": {
        "high_warning": 0.7,
    },
}


# -----------------------------
# Trend analysis
# -----------------------------
def compute_trends(recent_df: pd.DataFrame, horizon_hours: float = 6.0) -> dict:
    recent_df = recent_df.sort_values("timestamp").reset_index(drop=True)
    if len(recent_df) < 2:
        return {}

    t_end = recent_df["timestamp"].max()
    window_start = t_end - pd.Timedelta(hours=horizon_hours)
    window_df = recent_df[recent_df["timestamp"] >= window_start]
    if len(window_df) < 2:
        window_df = recent_df.tail(2)

    trends = {}

    for col in ["temperature", "pH", "turbidity_proxy", "predicted_do", "predicted_nh3"]:
        arr = window_df[col].values.astype(float)
        start = arr[0]
        end   = arr[-1]
        delta = end - start

        x = np.arange(len(arr))
        slope = np.polyfit(x, arr, 1)[0] if len(arr) > 1 else 0.0

        trends[col] = {
            "start": float(start),
            "end": float(end),
            "delta": float(delta),
            "slope": float(slope),
            "n_points": int(len(arr)),
        }

    return trends


# -----------------------------
# Explain current risk
# -----------------------------
def explain_current_risk(recent_df: pd.DataFrame, horizon_hours: float = 6.0) -> dict:
    recent_df = recent_df.sort_values("timestamp").reset_index(drop=True)
    last = recent_df.iloc[-1]

    # 1) Risk classifier
    X_risk = prepare_risk_features_from_row(last)
    proba  = risk_model.predict_proba(X_risk)[0]
    label_argmax = get_risk_label_from_proba(proba)

    # apply HIGH threshold: if P(HIGH) > threshold → label HIGH
    if "HIGH" in model_class_order:
        idx_high = model_class_order.tolist().index("HIGH")
        p_high   = float(proba[idx_high])
    else:
        p_high   = 0.0

    high_thr = risk_thresholds.get("HIGH", 0.3)
    final_label = "HIGH" if p_high >= high_thr else label_argmax

    # 2) Trends
    trends = compute_trends(recent_df, horizon_hours=horizon_hours)

    # 3) Factors
    factors: list[tuple[str, str]] = []

    # DO
    do = float(last["predicted_do"])
    do_thr = PARAM_THRESHOLDS["predicted_do"]
    if do < do_thr["low_critical"]:
        factors.append(("PRIMARY", f"Dissolved oxygen critically low ({do:.2f} mg/L)."))
    elif do < do_thr["low_warning"]:
        factors.append(("PRIMARY", f"Dissolved oxygen in risky zone ({do:.2f} mg/L)."))

    # NH3
    nh3 = float(last["predicted_nh3"])
    nh3_thr = PARAM_THRESHOLDS["predicted_nh3"]
    if nh3 > nh3_thr["high_critical"]:
        factors.append(("PRIMARY", f"Ammonia critically high ({nh3:.3f} mg/L)."))
    elif nh3 > nh3_thr["high_warning"]:
        factors.append(("CONTRIBUTING", f"Ammonia elevated ({nh3:.3f} mg/L)."))

    # Temperature
    temp = float(last["temperature"])
    t_thr = PARAM_THRESHOLDS["temperature"]
    if temp > t_thr["high"]:
        factors.append(("CONTRIBUTING", f"Water temperature high ({temp:.1f} °C)."))
    elif temp < t_thr["low"]:
        factors.append(("CONTRIBUTING", f"Water temperature low ({temp:.1f} °C)."))

    # pH
    ph = float(last["pH"])
    ph_thr = PARAM_THRESHOLDS["pH"]
    if ph < ph_thr["low"] or ph > ph_thr["high"]:
        factors.append(("CONTRIBUTING", f"pH out of comfort range ({ph:.2f})."))

    # Turbidity
    turb = float(last["turbidity_proxy"])
    turb_thr = PARAM_THRESHOLDS["turbidity_proxy"]["high_warning"]
    if turb > turb_thr:
        factors.append(("SECONDARY", f"Turbidity elevated (proxy={turb:.3f})."))

    # Trend-based
    for col, info in trends.items():
        delta = info["delta"]
        npts  = info["n_points"]
        if col == "predicted_do" and delta < -0.5:
            factors.append(("PRIMARY", f"Dissolved oxygen dropped {delta:.2f} mg/L over last {npts} readings."))
        if col == "predicted_nh3" and delta > 0.1:
            factors.append(("CONTRIBUTING", f"Ammonia rising by {delta:.3f} mg/L in recent window."))

    explanation = {
        "risk_label": final_label,
        "risk_label_argmax": label_argmax,
        "risk_proba": {cls: float(p) for cls, p in zip(model_class_order, proba)},
        "p_high": p_high,
        "high_threshold": high_thr,
        "factors": factors,
    }

        # 5) Fallback factor if nothing was detected
    if not factors:
        factors.append((
            "INFO",
            "No strong stressors detected based on thresholds; "
            "risk level is driven by the model's learned patterns from historical data."
        ))

    return explanation



# -----------------------------
# Recommendation engine
# -----------------------------
RECOMMENDATION_RULES = {
    "PRIMARY_DO_LOW": {
        "trigger": lambda ctx: ctx["last"]["predicted_do"] < PARAM_THRESHOLDS["predicted_do"]["low_warning"],
        "immediate": [
            "Increase aeration immediately.",
            "Avoid feeding until dissolved oxygen stabilizes.",
        ],
        "investigation": [
            "Check aerator performance and power supply.",
            "Observe fish behavior near the surface.",
        ],
        "preventive": [
            "Schedule DO measurements at dawn and late afternoon daily.",
            "Optimize feeding schedule to reduce excess organic load.",
        ],
    },
    "PRIMARY_NH3_HIGH": {
        "trigger": lambda ctx: ctx["last"]["predicted_nh3"] > PARAM_THRESHOLDS["predicted_nh3"]["high_warning"],
        "immediate": [
            "Reduce feeding by 50% for the next 24 hours.",
            "Increase partial water exchange if possible.",
        ],
        "investigation": [
            "Check for uneaten feed and sludge accumulation.",
            "Review recent changes in feeding or stocking density.",
        ],
        "preventive": [
            "Implement regular sludge removal.",
            "Align stocking densities with pond capacity.",
        ],
    },
    "TEMP_OUT_OF_RANGE": {
        "trigger": lambda ctx: (
            ctx["last"]["temperature"] > PARAM_THRESHOLDS["temperature"]["high"]
            or ctx["last"]["temperature"] < PARAM_THRESHOLDS["temperature"]["low"]
        ),
        "immediate": [
            "Avoid major feeding during hottest hours of the day.",
        ],
        "investigation": [
            "Check shading, water depth, and inflow temperature.",
        ],
        "preventive": [
            "Plan partial water exchange during cooler hours.",
            "Increase shading structures if persistent high temperatures.",
        ],
    },
}


def _dedup(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def generate_recommendations(recent_df: pd.DataFrame, explanation: dict) -> dict:
    recent_df = recent_df.sort_values("timestamp").reset_index(drop=True)
    last = recent_df.iloc[-1].to_dict()
    ctx = {"last": last, "explanation": explanation}

    immediate = []
    investigation = []
    preventive = []

    for name, rule in RECOMMENDATION_RULES.items():
        try:
            if rule["trigger"](ctx):
                immediate.extend(rule.get("immediate", []))
                investigation.extend(rule.get("investigation", []))
                preventive.extend(rule.get("preventive", []))
        except Exception as e:
            print(f"Rule {name} error: {e}")

        # Fallback: if nothing triggered, provide baseline recs based on risk label
    risk_label = explanation.get("risk_label", "LOW")

    if not immediate and not investigation and not preventive:
        if risk_label == "LOW":
            preventive = [
                "Maintain current aeration and feeding schedule.",
                "Log sensor readings at least twice per day.",
            ]
        elif risk_label == "MEDIUM":
            immediate = [
                "Increase observation frequency for the next 24 hours.",
            ]
            investigation = [
                "Check for any changes in feeding, stocking, or weather.",
            ]
            preventive = [
                "Plan a routine water quality check (DO, NH3) within the next 48 hours.",
            ]
        elif risk_label == "HIGH":
            immediate = [
                "Increase aeration immediately and avoid feeding until conditions stabilize.",
            ]
            investigation = [
                "Inspect fish behavior (gasping at surface) and verify aerator functioning.",
            ]
            preventive = [
                "Review stocking density and feeding rates to reduce long-term stress.",
            ]

    return {
        "immediate": _dedup(immediate),
        "investigation": _dedup(investigation),
        "preventive": _dedup(preventive),
    
    }


# -----------------------------
# Main API for external use
# -----------------------------
def build_decision_support_payload(recent_df: pd.DataFrame) -> dict:
    """
    recent_df must have >= HISTORY_STEPS rows and columns:
      ['timestamp','temperature','pH','turbidity_proxy','predicted_do','predicted_nh3']
    """
    recent_df = recent_df.sort_values("timestamp").reset_index(drop=True)
    assert len(recent_df) >= HISTORY_STEPS, f"Need >= {HISTORY_STEPS} rows, got {len(recent_df)}"

    explanation     = explain_current_risk(recent_df, horizon_hours=6.0)
    forecast        = predict_future_risk(recent_df)
    recommendations = generate_recommendations(recent_df, explanation)

    max_forecast = max(forecast.values())
    current_label = explanation["risk_label"]

    if current_label == "HIGH" or max_forecast > 0.7:
        alert_level = "CRITICAL"
    elif max_forecast > 0.4:
        alert_level = "WARNING"
    else:
        alert_level = "INFO"

    payload = {
        "current_risk": explanation,
        "forecast": forecast,
        "recommendations": recommendations,
        "alert_level": alert_level,
        "latest_timestamp": str(recent_df["timestamp"].max()),
    }
    return payload


if __name__ == "__main__":
    print("Decision support module loaded.")
    print("SENSORS:", SENSORS)
    print("HORIZONS:", HORIZONS)
    print("Risk model version:", risk_version)
'''

