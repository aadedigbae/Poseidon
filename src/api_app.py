"""
api_app.py

FastAPI application for Poseidon decision-support.
UPDATED: Runs analysis EVERY 15 seconds (after first 12 readings minimum).

NEW FEATURES:
- Automatic analysis after minimum readings collected
- Background processing (ESP32 never waits)
- All results saved to 4 CSV files automatically:
  * readings.csv
  * risk_predictions.csv
  * forecasts.csv
  * actions_log.csv
"""

import sys
import traceback
from datetime import datetime
from typing import List, Any, Dict
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd

# Add the parent directory to the Python path to handle imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Try to import with better error handling
try:
    from soft_sensors_runtime import enrich_with_virtual_sensors
    from decision_support import (
        HISTORY_STEPS,
        build_decision_support_payload,
        explain_current_risk,
        predict_future_risk,
    )
    import storage
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {Path.cwd()}")
    print(f"ROOT directory: {ROOT}")
    print(f"Files in ROOT: {list(ROOT.iterdir())}")
    raise

app = FastAPI(
    title="Poseidon Aquaculture Decision Support API",
    version="2.0.0",
    description="Real-time ML engine with automated analysis every 15 seconds.",
)


# ============================================================================
# CONFIGURATION - ANALYSIS EVERY 15 SECONDS (AFTER MINIMUM DATA)
# ============================================================================

ANALYSIS_CONFIG = {
    # Run analysis on EVERY reading (not every 720!)
    # Set to 1 to run every single reading after minimum is met
    "run_every_n_readings": 1,
    
    # Minimum readings needed before first analysis
    # This uses HISTORY_STEPS from your model bundle
    # If your models need 168 readings, use 168
    # If your models can work with 12, use 12
    "min_readings_required": max(12, HISTORY_STEPS),  # At least 12, or model requirement
    
    # Enable/disable auto-analysis
    "auto_analysis_enabled": True,
}

# Counter to track readings per pond
reading_counters = {}

print(f"\n{'='*60}")
print(f"ANALYSIS CONFIGURATION:")
print(f"  Min readings required: {ANALYSIS_CONFIG['min_readings_required']}")
print(f"  Run frequency: Every {ANALYSIS_CONFIG['run_every_n_readings']} reading(s)")
print(f"  HISTORY_STEPS from models: {HISTORY_STEPS}")
print(f"  Time until first analysis: ~{(ANALYSIS_CONFIG['min_readings_required'] * 15) / 60:.1f} minutes")
print(f"{'='*60}\n")


# ============================================================================
# BACKGROUND ANALYSIS FUNCTION (RUNS AFTER ESP32 GETS RESPONSE)
# ============================================================================

def should_run_analysis(pond_id: str) -> bool:
    """
    Check if we should run analysis for this pond.
    Returns True on every reading after minimum is met.
    """
    global reading_counters
    
    if pond_id not in reading_counters:
        reading_counters[pond_id] = 0
    
    reading_counters[pond_id] += 1
    
    # Always run if we've met the minimum (since run_every_n_readings = 1)
    if reading_counters[pond_id] >= ANALYSIS_CONFIG["min_readings_required"]:
        if reading_counters[pond_id] % ANALYSIS_CONFIG["run_every_n_readings"] == 0:
            return True
    
    return False


def try_run_analysis(pond_id: str):
    """
    Background task: Run complete analysis pipeline.
    
    Steps:
    1. Fetch last HISTORY_STEPS readings from storage
    2. Run risk classifier
    3. Run forecast models
    4. Generate recommendations
    5. Save all results to CSV files
    
    This runs AFTER ESP32 gets response, so sensors don't wait.
    """
    try:
        print(f"\n{'='*60}")
        print(f"[{pond_id}] Starting automatic analysis...")
        print(f"{'='*60}")
        
        # STEP 1: Get recent readings from storage
        min_required = ANALYSIS_CONFIG["min_readings_required"]
        df = storage.get_recent_readings(pond_id, min_required)
        
        # STEP 2: Check if we have enough data
        if len(df) < min_required:
            needed = min_required - len(df)
            seconds_left = needed * 15
            minutes_left = seconds_left / 60
            print(f"[{pond_id}] ⏳ Not enough readings yet: {len(df)}/{min_required}")
            print(f"[{pond_id}]    Need {needed} more readings (~{minutes_left:.1f} minutes)")
            return
        
        # STEP 3: Clean and prepare data
        if "timestamp" not in df.columns:
            print(f"[{pond_id}] ❌ ERROR: No timestamp column")
            return
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        
        if len(df) < min_required:
            print(f"[{pond_id}] ❌ Not enough valid readings after cleaning: {len(df)}/{min_required}")
            return
        
        # STEP 4: Run the full analysis pipeline
        print(f"[{pond_id}] 🧠 Running AI analysis...")
        print(f"[{pond_id}]    Using last {len(df)} readings")
        print(f"[{pond_id}]    Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        payload = build_decision_support_payload(df)
        
        # STEP 5: Persist results to storage
        persist_decision_support_event(pond_id, df, payload, include_readings=False)
        
        # STEP 6: Extract key results
        alert_level = payload.get("alert_level", "UNKNOWN")
        current_risk = payload.get("current_risk", {})
        risk_label = current_risk.get("risk_label", "UNKNOWN")
        p_high = current_risk.get("p_high", 0.0)
        forecast = payload.get("forecast", {})
        
        # STEP 7: Log results
        print(f"\n[{pond_id}] ✅ Analysis complete!")
        print(f"[{pond_id}]    Alert Level: {alert_level}")
        print(f"[{pond_id}]    Current Risk: {risk_label} (p_high={p_high:.3f})")
        print(f"[{pond_id}]    Forecast +1h: {forecast.get('+1h', 0.0):.3f}")
        print(f"[{pond_id}]    Forecast +6h: {forecast.get('+6h', 0.0):.3f}")
        print(f"[{pond_id}]    Forecast +24h: {forecast.get('+24h', 0.0):.3f}")
        
        # STEP 8: Handle critical alerts
        if alert_level == "CRITICAL":
            print(f"\n[{pond_id}] 🚨 CRITICAL ALERT TRIGGERED!")
            recs = payload.get("recommendations", {})
            immediate = recs.get("immediate", [])
            
            print(f"[{pond_id}] Immediate actions required:")
            for i, action in enumerate(immediate, 1):
                print(f"[{pond_id}]    {i}. {action}")
            
            # TODO: Add SMS/Email alerts here
            # send_critical_alert(pond_id, payload)
        
        elif alert_level == "WARNING":
            print(f"[{pond_id}] ⚠️  WARNING: Monitor closely")
        
        else:
            print(f"[{pond_id}] ✅ All systems normal")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n[{pond_id}] ❌ ERROR in background analysis:")
        print(f"[{pond_id}] {str(e)}")
        traceback.print_exc()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def persist_decision_support_event(
    pond_id: str,
    df: pd.DataFrame,
    payload: Dict[str, Any],
    include_readings: bool = False,
) -> None:
    """
    Save the outputs of a decision-support event into storage.
    Creates 3 CSV files (or adds rows if they exist):
    - risk_predictions.csv
    - forecasts.csv
    - actions_log.csv
    """
    try:
        # 1) Optional: save readings (usually not needed since already saved)
        if include_readings:
            for _, row in df.iterrows():
                rdict = row.to_dict()
                if isinstance(rdict.get("timestamp"), pd.Timestamp):
                    rdict["timestamp"] = rdict["timestamp"].to_pydatetime()
                storage.save_reading(pond_id, rdict)

        # 2) Determine latest timestamp
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column to persist event.")

        latest_ts = df["timestamp"].max()
        if isinstance(latest_ts, pd.Timestamp):
            latest_ts = latest_ts.to_pydatetime()

        # 3) Risk snapshot
        current_risk = payload.get("current_risk", {})
        risk_proba = current_risk.get("risk_proba", {})

        risk_row = {
            "timestamp": latest_ts,
            "temperature": float(df.iloc[-1]["temperature"]),
            "pH": float(df.iloc[-1]["pH"]),
            "turbidity_proxy": float(df.iloc[-1]["turbidity_proxy"]),
            "predicted_do": float(df.iloc[-1]["predicted_do"]),
            "predicted_nh3": float(df.iloc[-1]["predicted_nh3"]),
            "risk_label": current_risk.get("risk_label"),
            "risk_label_argmax": current_risk.get("risk_label_argmax"),
            "prob_HIGH": float(risk_proba.get("HIGH", 0.0)),
            "prob_MEDIUM": float(risk_proba.get("MEDIUM", 0.0)),
            "prob_LOW": float(risk_proba.get("LOW", 0.0)),
            "p_high": float(current_risk.get("p_high", 0.0)),
            "high_threshold": float(current_risk.get("high_threshold", 0.0)),
            "model_version": "model1_risk_classifier_v1",
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        storage.save_risk_prediction(pond_id, risk_row)

        # 4) Forecast snapshot
        forecast = payload.get("forecast", {})
        current_p_high = float(current_risk.get("p_high", 0.0))
        risk_trend = compute_risk_trend(current_p_high, forecast)

        forecast_row = {
            "timestamp": latest_ts,
            "forecast_+1h": float(forecast.get("+1h", 0.0)),
            "forecast_+6h": float(forecast.get("+6h", 0.0)),
            "forecast_+24h": float(forecast.get("+24h", 0.0)),
            "forecast_+3d": float(forecast.get("+3d", 0.0)),
            "risk_trend": risk_trend,
            "model_version": "model3_forecast_hybrid_v1",
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        storage.save_forecast(pond_id, forecast_row)

        # 5) Actions / recommendations snapshot
        recs = payload.get("recommendations", {})
        actions_row = {
            "timestamp": latest_ts,
            "alert_level": payload.get("alert_level"),
            "immediate_actions": recs.get("immediate", []),
            "investigation_actions": recs.get("investigation", []),
            "preventive_actions": recs.get("preventive", []),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        storage.save_actions(pond_id, actions_row)
        
    except Exception as e:
        print(f"Error in persist_decision_support_event: {e}")
        print(traceback.format_exc())


def compute_risk_trend(current_p_high: float, forecast: Dict[str, float], tol: float = 0.05) -> str:
    """
    Compute qualitative risk trend label.
    """
    if not forecast:
        return "UNKNOWN"

    future_vals = list(forecast.values())
    avg_future = sum(future_vals) / len(future_vals)
    delta = avg_future - current_p_high

    if delta > tol:
        return "INCREASING"
    elif delta < -tol:
        return "DECREASING"
    else:
        return "STABLE"


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SensorReading(BaseModel):
    timestamp: datetime = Field(..., description="ISO timestamp of reading")
    temperature: float = Field(..., description="Water temperature in °C")
    pH: float = Field(..., description="pH")
    turbidity_proxy: float = Field(..., description="Turbidity proxy (0–1)")
    predicted_do: float = Field(..., description="Virtual DO (mg/L)")
    predicted_nh3: float = Field(..., description="Virtual NH3 (mg/L)")


class DecisionSupportRequest(BaseModel):
    pond_id: str = Field(..., description="Pond or site identifier")
    readings: List[SensorReading] = Field(..., description="Recent readings")


class DecisionSupportResponse(BaseModel):
    pond_id: str
    payload: Dict[str, Any]


class RawSensorReading(BaseModel):
    timestamp: datetime = Field(..., description="ISO timestamp of reading")
    temperature: float = Field(..., description="Water temperature in °C")
    pH: float = Field(..., description="pH value")
    turbidity_proxy: float = Field(..., description="Turbidity proxy (0–1)")


class IngestReadingRequest(BaseModel):
    pond_id: str = Field(..., description="Pond or site identifier")
    reading: RawSensorReading = Field(..., description="Single raw reading from ESP32")


class IngestReadingResponse(BaseModel):
    pond_id: str
    stored: bool
    enriched_reading: Dict[str, Any]


def readings_to_dataframe(readings: List[SensorReading]) -> pd.DataFrame:
    data = [r.model_dump() for r in readings]
    df = pd.DataFrame(data)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
def health():
    """Health check endpoint with detailed status."""
    try:
        hours_needed = (ANALYSIS_CONFIG['min_readings_required'] * 15) / 3600
        
        status_info = {
            "status": "ok",
            "message": "Poseidon decision support API is running.",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "root_path": str(ROOT),
            "current_dir": str(Path.cwd()),
            "python_version": sys.version,
            "storage_backend": getattr(storage, 'BACKEND', 'unknown'),
            "auto_analysis_enabled": ANALYSIS_CONFIG["auto_analysis_enabled"],
            "analysis_frequency": f"Every {ANALYSIS_CONFIG['run_every_n_readings']} reading(s) after minimum",
            "min_readings_required": ANALYSIS_CONFIG["min_readings_required"],
            "time_span_required_hours": round(hours_needed, 2),
            "history_steps_from_models": HISTORY_STEPS,
        }
        return status_info
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/ingest_reading", response_model=IngestReadingResponse)
def ingest_reading(req: IngestReadingRequest, background_tasks: BackgroundTasks):
    """
    Enhanced endpoint for ESP32 to send a single raw reading.
    
    NEW: Auto-triggers analysis EVERY reading after minimum data collected.
    The ESP32 gets an immediate response (~100ms) and doesn't wait for analysis.
    
    Flow:
    1. Enrich raw data with virtual DO and NH3
    2. Save to readings.csv
    3. Check if we should run analysis
    4. If yes, trigger background analysis
    5. Return immediately to ESP32
    """
    try:
        pond_id = req.pond_id
        raw = req.reading.model_dump()
        
        # Step 1: Enrich with virtual sensors (DO and NH3)
        enriched = enrich_with_virtual_sensors(raw)
        
        # Step 2: Save the reading to storage (readings.csv)
        storage.save_reading(pond_id, enriched)
        
        # Step 3: Check if we should run analysis
        if ANALYSIS_CONFIG["auto_analysis_enabled"] and should_run_analysis(pond_id):
            count = reading_counters.get(pond_id, 0)
            print(f"[{pond_id}] 🔄 Triggering background analysis (reading #{count})")
            background_tasks.add_task(try_run_analysis, pond_id)
        
        # Step 4: Return immediately to ESP32
        return IngestReadingResponse(
            pond_id=pond_id,
            stored=True,
            enriched_reading=enriched,
        )
        
    except Exception as e:
        print(f"[{pond_id}] ❌ ERROR in ingest_reading: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing reading: {str(e)}")


@app.post("/decision_support", response_model=DecisionSupportResponse, summary="Full decision-support payload")
def decision_support(req: DecisionSupportRequest):
    """Full decision-support payload with enhanced error handling."""
    pond_id = req.pond_id

    if len(req.readings) < HISTORY_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {HISTORY_STEPS} readings, got {len(req.readings)}.",
        )

    # 1) Convert readings -> DataFrame
    df = pd.DataFrame([r.model_dump() for r in req.readings])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 2) Build decision-support payload (model logic)
    try:
        payload = build_decision_support_payload(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 3) Persist event (include_readings=True because client sent them)
    persist_decision_support_event(pond_id, df, payload, include_readings=True)

    # 4) Return the payload
    return DecisionSupportResponse(
        pond_id=pond_id,
        payload=payload,
    )


@app.get(
    "/ponds/{pond_id}/decision_support_from_history",
    response_model=DecisionSupportResponse,
    summary="Decision support using recent history from storage"
)
def decision_support_from_history(pond_id: str, n_steps: int | None = None):
    """
    Fetch the last n_steps readings for this pond from storage,
    run the decision-support pipeline, persist risk/forecast/actions, and
    return the same payload structure as POST /decision_support.
    """
    # 1) Determine how many steps to use
    steps = n_steps or HISTORY_STEPS

    # 2) Pull recent readings from storage
    df = storage.get_recent_readings(pond_id, steps)

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No readings found for pond_id='{pond_id}'.",
        )

    # Ensure proper sorting & timestamp parsing
    if "timestamp" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="Stored readings are missing 'timestamp' column.",
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if len(df) < HISTORY_STEPS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough readings for decision support. "
                f"Need at least {HISTORY_STEPS}, found {len(df)}."
            ),
        )

    # 3) Run decision-support pipeline
    try:
        payload = build_decision_support_payload(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 4) Persist risk/forecast/actions (readings already stored)
    persist_decision_support_event(pond_id, df, payload, include_readings=False)

    # 5) Return response
    return DecisionSupportResponse(
        pond_id=pond_id,
        payload=payload,
    )


@app.post("/current_risk", summary="Current risk only (no forecast)")
def current_risk(readings: List[SensorReading]):
    """Analyze current risk without forecast."""
    if not readings:
        raise HTTPException(status_code=400, detail="At least one reading is required.")

    df = readings_to_dataframe(readings)

    try:
        expl = explain_current_risk(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "latest_timestamp": str(df["timestamp"].max()),
        "current_risk": expl,
    }


@app.post("/forecast_only", summary="Forecast only using hybrid model")
def forecast_only(readings: List[SensorReading]):
    """Generate forecast without current risk analysis."""
    if len(readings) < HISTORY_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {HISTORY_STEPS} readings, got {len(readings)}.",
        )

    df = readings_to_dataframe(readings)

    try:
        fc = predict_future_risk(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "latest_timestamp": str(df["timestamp"].max()),
        "forecast": fc,
    }


# ============================================================================
# MONITORING ENDPOINTS (FOR TESTING & STATUS)
# ============================================================================

@app.post("/ponds/{pond_id}/force_analysis")
def force_analysis(pond_id: str, background_tasks: BackgroundTasks):
    """
    Manually force an analysis run (useful for testing).
    Returns immediately, analysis runs in background.
    """
    background_tasks.add_task(try_run_analysis, pond_id)
    return {
        "status": "triggered",
        "pond_id": pond_id,
        "message": "Analysis started in background. Check console logs for results."
    }


@app.get("/ponds/{pond_id}/analysis_status")
def analysis_status(pond_id: str):
    """
    Check analysis status for a pond.
    Shows counter status and time estimates.
    """
    reading_count = reading_counters.get(pond_id, 0)
    min_required = ANALYSIS_CONFIG["min_readings_required"]
    
    if reading_count < min_required:
        # Still collecting initial data
        needed = min_required - reading_count
        seconds_left = needed * 15
        hours_left = seconds_left / 3600
        
        return {
            "pond_id": pond_id,
            "status": "collecting_data",
            "reading_counter": reading_count,
            "min_readings_required": min_required,
            "readings_needed": needed,
            "time_remaining": {
                "seconds": seconds_left,
                "minutes": round(seconds_left / 60, 1),
                "hours": round(hours_left, 2),
            },
            "message": f"Collecting data: {reading_count}/{min_required} readings"
        }
    else:
        # Analysis is running
        return {
            "pond_id": pond_id,
            "status": "active",
            "reading_counter": reading_count,
            "min_readings_required": min_required,
            "analysis_frequency": f"Every {ANALYSIS_CONFIG['run_every_n_readings']} reading(s) (~15 seconds)",
            "auto_analysis_enabled": ANALYSIS_CONFIG["auto_analysis_enabled"],
            "message": "Analysis running automatically"
        }


@app.post("/config/update_analysis_frequency")
def update_analysis_frequency(new_frequency: int):
    """
    Dynamically change how often analysis runs.
    
    Examples:
    - 1: Every reading (default for real-time)
    - 4: Every 4th reading (~1 minute)
    - 12: Every 12th reading (~3 minutes)
    """
    if new_frequency < 1:
        raise HTTPException(status_code=400, detail="Frequency must be >= 1")
    
    old_frequency = ANALYSIS_CONFIG["run_every_n_readings"]
    ANALYSIS_CONFIG["run_every_n_readings"] = new_frequency
    
    seconds_old = old_frequency * 15
    seconds_new = new_frequency * 15
    
    return {
        "status": "updated",
        "old_frequency": old_frequency,
        "new_frequency": new_frequency,
        "old_interval_seconds": seconds_old,
        "new_interval_seconds": seconds_new,
        "message": f"Analysis will now run every {new_frequency} reading(s) (~{seconds_new} seconds)"
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    hours_needed = (ANALYSIS_CONFIG['min_readings_required'] * 15) / 3600
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║        Poseidon Aquaculture Decision Support API             ║
    ║               AUTOMATED REAL-TIME ANALYSIS                    ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    CONFIGURATION:
    - Analysis runs: Every {ANALYSIS_CONFIG['run_every_n_readings']} reading(s) after minimum
    - Min data needed: {ANALYSIS_CONFIG['min_readings_required']} readings (~{hours_needed:.1f} hours)
    - Model HISTORY_STEPS: {HISTORY_STEPS}
    - Background processing: ✅ (ESP32 doesn't wait)
    
    WHAT HAPPENS:
    1. ESP32 sends data every 15 seconds
    2. Saved to readings.csv
    3. After {ANALYSIS_CONFIG['min_readings_required']} readings: Full analysis starts
    4. Analysis runs every 15 seconds (or configured interval)
    5. Results saved automatically:
       - risk_predictions.csv
       - forecasts.csv
       - actions_log.csv
    
    ALL 4 FILES UPDATE AUTOMATICALLY! ✅
    
    Starting server on http://0.0.0.0:8000
    Docs at http://localhost:8000/docs
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


'''
"""
api_app.py

FastAPI application exposing Poseidon decision-support endpoints.
"""

from datetime import datetime
from typing import List, Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .soft_sensors_runtime import enrich_with_virtual_sensors
import pandas as pd

from .decision_support import (
    HISTORY_STEPS,
    build_decision_support_payload,
    explain_current_risk,
    predict_future_risk,  # from runtime_forecast (re-exported or import directly)
)

app = FastAPI(
    title="Poseidon Aquaculture Decision Support API",
    version="1.0.0",
    description="Hybrid ML + AI engine for real-time water quality risk & recommendations.",
)

from . import storage  # <--- NEW: our storage wrapper (local/Firebase later)

def persist_decision_support_event(
    pond_id: str,
    df: pd.DataFrame,
    payload: Dict[str, Any],
    include_readings: bool = False,
) -> None:
    """
    Save the outputs of a decision-support event into storage.

    - df: DataFrame of readings (must include timestamp, temperature, pH,
          turbidity_proxy, predicted_do, predicted_nh3).
    - payload: output of build_decision_support_payload(df).
    - include_readings: if True, will also save each reading via storage.save_reading.
      For POST /decision_support (client sends readings) we often do this.
      For history-based endpoint (data already in Firestore), we can skip.
    """
    # 1) Optional: save readings
    if include_readings:
        for _, row in df.iterrows():
            rdict = row.to_dict()
            # Make sure timestamp is JSON/Firestore-friendly
            if isinstance(rdict.get("timestamp"), pd.Timestamp):
                rdict["timestamp"] = rdict["timestamp"].to_pydatetime()
            storage.save_reading(pond_id, rdict)

    # 2) Determine latest timestamp
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column to persist event.")

    latest_ts = df["timestamp"].max()
    if isinstance(latest_ts, pd.Timestamp):
        latest_ts = latest_ts.to_pydatetime()

    # 3) Risk snapshot
    current_risk = payload.get("current_risk", {})
    risk_proba   = current_risk.get("risk_proba", {})

    risk_row = {
        "timestamp": latest_ts,
        "temperature": float(df.iloc[-1]["temperature"]),
        "pH": float(df.iloc[-1]["pH"]),
        "turbidity_proxy": float(df.iloc[-1]["turbidity_proxy"]),
        "predicted_do": float(df.iloc[-1]["predicted_do"]),
        "predicted_nh3": float(df.iloc[-1]["predicted_nh3"]),

        "risk_label": current_risk.get("risk_label"),
        "risk_label_argmax": current_risk.get("risk_label_argmax"),
        "prob_HIGH": float(risk_proba.get("HIGH", 0.0)),
        "prob_MEDIUM": float(risk_proba.get("MEDIUM", 0.0)),
        "prob_LOW": float(risk_proba.get("LOW", 0.0)),
        "p_high": float(current_risk.get("p_high", 0.0)),
        "high_threshold": float(current_risk.get("high_threshold", 0.0)),
        "model_version": "model1_risk_classifier_v1",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    storage.save_risk_prediction(pond_id, risk_row)

    # 4) Forecast snapshot
    forecast = payload.get("forecast", {})
    current_p_high = float(current_risk.get("p_high", 0.0))
    risk_trend = compute_risk_trend(current_p_high, forecast)

    forecast_row = {
        "timestamp": latest_ts,
        "forecast_+1h": float(forecast.get("+1h", 0.0)),
        "forecast_+6h": float(forecast.get("+6h", 0.0)),
        "forecast_+24h": float(forecast.get("+24h", 0.0)),
        "forecast_+3d": float(forecast.get("+3d", 0.0)),
        "risk_trend": risk_trend,
        "model_version": "model3_forecast_hybrid_v1",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    storage.save_forecast(pond_id, forecast_row)

    # 5) Actions / recommendations snapshot
    recs = payload.get("recommendations", {})
    actions_row = {
        "timestamp": latest_ts,
        "alert_level": payload.get("alert_level"),
        "immediate_actions": recs.get("immediate", []),
        "investigation_actions": recs.get("investigation", []),
        "preventive_actions": recs.get("preventive", []),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    storage.save_actions(pond_id, actions_row)


def compute_risk_trend(current_p_high: float, forecast: Dict[str, float], tol: float = 0.05) -> str:
    """
    Compute qualitative risk trend label ("INCREASING", "STABLE", "DECREASING")
    based on current P(HIGH) vs average future P(HIGH).
    tol: tolerance range where we consider it "STABLE".
    """
    if not forecast:
        return "UNKNOWN"

    future_vals = list(forecast.values())
    avg_future = sum(future_vals) / len(future_vals)

    delta = avg_future - current_p_high

    if delta > tol:
        return "INCREASING"
    elif delta < -tol:
        return "DECREASING"
    else:
        return "STABLE"
    

# -------------------------------------------------------------------
# Pydantic models (request / response schemas)
# -------------------------------------------------------------------

class SensorReading(BaseModel):
    timestamp: datetime = Field(..., description="ISO timestamp of reading")
    temperature: float = Field(..., description="Water temperature in °C")
    pH: float = Field(..., description="pH")
    turbidity_proxy: float = Field(..., description="Turbidity proxy (0–1)")
    predicted_do: float = Field(..., description="Virtual DO (mg/L)")
    predicted_nh3: float = Field(..., description="Virtual NH3 (mg/L)")


class DecisionSupportRequest(BaseModel):
    pond_id: str = Field(..., description="Pond or site identifier")
    readings: List[SensorReading] = Field(..., description="Recent readings, ordered or unordered")


# We can be loose with the response type and return Dict[str, Any],
# because build_decision_support_payload already builds a JSON-serializable dict.
class DecisionSupportResponse(BaseModel):
    pond_id: str
    payload: Dict[str, Any]

class RawSensorReading(BaseModel):
    timestamp: datetime = Field(..., description="ISO timestamp of reading")
    temperature: float = Field(..., description="Water temperature in °C")
    pH: float = Field(..., description="pH value")
    turbidity_proxy: float = Field(..., description="Turbidity proxy (0–1 or sensor units)")


class IngestReadingRequest(BaseModel):
    pond_id: str = Field(..., description="Pond or site identifier")
    reading: RawSensorReading = Field(..., description="Single raw reading from ESP32")


class IngestReadingResponse(BaseModel):
    pond_id: str
    stored: bool
    enriched_reading: Dict[str, Any]

# -------------------------------------------------------------------
# Helper: convert list[SensorReading] -> DataFrame
# -------------------------------------------------------------------

def readings_to_dataframe(readings: List[SensorReading]) -> pd.DataFrame:
    data = [r.model_dump() for r in readings]
    df = pd.DataFrame(data)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "message": "Poseidon decision support API is running.",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

@app.post("/ingest_reading", response_model=IngestReadingResponse, summary="Ingest one raw reading from ESP32")
def ingest_reading(req: IngestReadingRequest):
    """
    Endpoint for the ESP32 (or any edge device) to send a single raw reading:
      - timestamp
      - temperature
      - pH
      - turbidity_proxy

    Backend:
      1) Enriches with virtual DO and NH3.
      2) Stores the full reading via the storage layer.
      3) Returns the enriched reading.
    """
    pond_id = req.pond_id

    # 1) Convert the raw reading into a dict
    raw = req.reading.model_dump()

    # 2) Enrich with virtual sensors (DO, NH3)
    enriched = enrich_with_virtual_sensors(raw)

    # 3) Save into readings storage
    storage.save_reading(pond_id, enriched)

    # 4) Return to client
    return IngestReadingResponse(
        pond_id=pond_id,
        stored=True,
        enriched_reading=enriched,
    )


@app.post("/decision_support", response_model=DecisionSupportResponse, summary="Full decision-support payload")
def decision_support(req: DecisionSupportRequest):
    pond_id = req.pond_id

    if len(req.readings) < HISTORY_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {HISTORY_STEPS} readings, got {len(req.readings)}.",
        )

    # 1) Convert readings -> DataFrame
    df = pd.DataFrame([r.model_dump() for r in req.readings])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 2) Build decision-support payload (model logic)
    try:
        payload = build_decision_support_payload(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 3) Persist event (this time include_readings=True because client sent them)
    persist_decision_support_event(pond_id, df, payload, include_readings=True)

    """
    # ------------------------------------------------------------------
    # 3) STORAGE: save readings, risk, forecast, actions
    # ------------------------------------------------------------------

    # (a) Save each reading (already enriched: has predicted_do, predicted_nh3)
    for r in req.readings:
        storage.save_reading(pond_id, r.model_dump())

    # We'll use the very last reading timestamp as "event time"
    latest_ts = df["timestamp"].max()

    # (b) Save risk prediction snapshot
    current_risk = payload.get("current_risk", {})
    risk_proba   = current_risk.get("risk_proba", {})

    risk_row = {
        "timestamp": latest_ts,
        "temperature": float(df.iloc[-1]["temperature"]),
        "pH": float(df.iloc[-1]["pH"]),
        "turbidity_proxy": float(df.iloc[-1]["turbidity_proxy"]),
        "predicted_do": float(df.iloc[-1]["predicted_do"]),
        "predicted_nh3": float(df.iloc[-1]["predicted_nh3"]),

        "risk_label": current_risk.get("risk_label"),
        "risk_label_argmax": current_risk.get("risk_label_argmax"),
        "prob_HIGH": float(risk_proba.get("HIGH", 0.0)),
        "prob_MEDIUM": float(risk_proba.get("MEDIUM", 0.0)),
        "prob_LOW": float(risk_proba.get("LOW", 0.0)),
        "p_high": float(current_risk.get("p_high", 0.0)),
        "high_threshold": float(current_risk.get("high_threshold", 0.0)),
        "model_version": "model1_risk_classifier_v1",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    storage.save_risk_prediction(pond_id, risk_row)

    # (c) Save forecast snapshot
    forecast = payload.get("forecast", {})
    current_p_high = float(current_risk.get("p_high", 0.0))
    risk_trend = compute_risk_trend(current_p_high, forecast)

    forecast_row = {
        "timestamp": latest_ts,
        "forecast_+1h": float(forecast.get("+1h", 0.0)),
        "forecast_+6h": float(forecast.get("+6h", 0.0)),
        "forecast_+24h": float(forecast.get("+24h", 0.0)),
        "forecast_+3d": float(forecast.get("+3d", 0.0)),
        "risk_trend": risk_trend,
        "model_version": "model3_forecast_hybrid_v1",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    storage.save_forecast(pond_id, forecast_row)

    # (d) Save recommendations / alerts
    recs = payload.get("recommendations", {})
    actions_row = {
        "timestamp": latest_ts,
        "alert_level": payload.get("alert_level"),
        "immediate_actions": recs.get("immediate", []),
        "investigation_actions": recs.get("investigation", []),
        "preventive_actions": recs.get("preventive", []),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    storage.save_actions(pond_id, actions_row)
    """

    # ------------------------------------------------------------------
    # 4) Return the payload unchanged
    # ------------------------------------------------------------------
    return DecisionSupportResponse(
        pond_id=pond_id,
        payload=payload,
    )

@app.get(
    "/ponds/{pond_id}/decision_support_from_history",
    response_model=DecisionSupportResponse,
    summary="Decision support using recent history from storage"
)
def decision_support_from_history(pond_id: str, n_steps: int | None = None):
    """
    Fetch the last n_steps readings for this pond from storage (Firestore or local),
    run the decision-support pipeline, persist risk/forecast/actions, and
    return the same payload structure as POST /decision_support.

    - pond_id: ID of the pond (e.g. RWA-01)
    - n_steps: optional; defaults to HISTORY_STEPS if not provided.
    """
    # 1) Determine how many steps to use
    steps = n_steps or HISTORY_STEPS

    # 2) Pull recent readings from storage
    df = storage.get_recent_readings(pond_id, steps)

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No readings found for pond_id='{pond_id}'.",
        )

    # Ensure proper sorting & timestamp parsing
    if "timestamp" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="Stored readings are missing 'timestamp' column.",
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if len(df) < HISTORY_STEPS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough readings for decision support. "
                f"Need at least {HISTORY_STEPS}, found {len(df)}."
            ),
        )

    # 3) Run decision-support pipeline
    try:
        payload = build_decision_support_payload(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 4) Persist risk/forecast/actions (readings are already stored, so include_readings=False)
    persist_decision_support_event(pond_id, df, payload, include_readings=False)

    # 5) Return response
    return DecisionSupportResponse(
        pond_id=pond_id,
        payload=payload,
    )


#others
@app.post("/current_risk", summary="Current risk only (no forecast)")
def current_risk(readings: List[SensorReading]):
    if not readings:
        raise HTTPException(status_code=400, detail="At least one reading is required.")

    df = readings_to_dataframe(readings)

    try:
        expl = explain_current_risk(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "latest_timestamp": str(df["timestamp"].max()),
        "current_risk": expl,
    }


@app.post("/forecast_only", summary="Forecast only using hybrid model")
def forecast_only(readings: List[SensorReading]):
    if len(readings) < HISTORY_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {HISTORY_STEPS} readings, got {len(readings)}.",
        )

    df = readings_to_dataframe(readings)

    try:
        fc = predict_future_risk(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "latest_timestamp": str(df["timestamp"].max()),
        "forecast": fc,
    }
'''