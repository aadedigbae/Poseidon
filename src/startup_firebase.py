"""
startup_firebase.py

Small helper to run at container startup:
- Reads FIREBASE_SERVICE_ACCOUNT_JSON from environment (a JSON string).
- Writes it to the path in POSEIDON_FIREBASE_CREDENTIALS.
"""

from pathlib import Path
import os
import json


def write_service_account_file():
    json_str = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    cred_path = os.getenv("POSEIDON_FIREBASE_CREDENTIALS")

    if not cred_path:
        print("[startup_firebase] POSEIDON_FIREBASE_CREDENTIALS not set. Skipping.")
        return

    cred_file = Path(cred_path).expanduser().resolve()

    if not json_str:
        print("[startup_firebase] FIREBASE_SERVICE_ACCOUNT_JSON not set. "
              "No file will be written. If you use local storage, this is fine.")
        return

    # Create parent folder
    cred_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Validate JSON first
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("[startup_firebase] FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON:", e)
        return

    with cred_file.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"[startup_firebase] Wrote Firebase service account to {cred_file}")


if __name__ == "__main__":
    write_service_account_file()
