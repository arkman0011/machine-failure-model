from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Machine Failure API")

# Load everything once, at startup
model = joblib.load("machine_failure_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

request_count = 0

class MachineInput(BaseModel):
    machine_type: Literal["L", "M", "H"]
    air_temperature: float = Field(ge=295, le=305)
    process_temperature: float = Field(ge=305, le=315)
    rotational_speed: float = Field(ge=1100, le=3000)
    torque: float = Field(ge=0, le=80)
    tool_wear: float = Field(ge=0, le=260)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metrics")
def metrics():
    return {"total_prediction_requests": request_count}


@app.post("/predict")
def predict(data: MachineInput):
    global request_count
    request_count += 1

    # 1. Engineer the same two features your notebook created
    temperature_difference = data.process_temperature - data.air_temperature
    power = data.torque * data.rotational_speed * 2 * np.pi / 60

    # 2. Build a one-row dict with raw + engineered features
    #    (column names match your cleaned names from Cell 14 — no brackets)
    row = {
        "Air temperature K": data.air_temperature,
        "Process temperature K": data.process_temperature,
        "Rotational speed rpm": data.rotational_speed,
        "Torque Nm": data.torque,
        "Tool wear min": data.tool_wear,
        "Temperature difference": temperature_difference,
        "Power W": power,
        # 3. One-hot encode Type manually (same as pd.get_dummies did)
        "Type_H": 1 if data.machine_type == "H" else 0,
        "Type_L": 1 if data.machine_type == "L" else 0,
        "Type_M": 1 if data.machine_type == "M" else 0,
    }

    input_df = pd.DataFrame([row])

    # 4. Reorder columns to match training exactly
    input_df = input_df[feature_columns]

    # 5. Scale using the SAME fitted scaler (transform, not fit_transform)
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_columns)

    # 6. Predict using the loaded model
    prediction = int(model.predict(input_scaled_df)[0])
    result = "Machine Failure" if prediction == 1 else "No Machine Failure"

    return {"prediction": prediction, "result": result}
#python -m uvicorn main:app --reload