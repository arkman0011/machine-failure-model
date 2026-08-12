from pathlib import Path
from threading import Lock
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Machine Failure Prediction API",
    description="Predicts whether industrial equipment is likely to fail.",
    version="1.0.0",
)


# --------------------------------------------------
# Load trained model files
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "machine_failure_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = list(joblib.load(FEATURE_COLUMNS_PATH))


# --------------------------------------------------
# Prediction-request counter
# --------------------------------------------------

request_count = 0
request_lock = Lock()


# --------------------------------------------------
# Request-data validation
# --------------------------------------------------

class MachineInput(BaseModel):
    machine_type: Literal["L", "M", "H"] = Field(
        description="Machine quality type: L, M, or H"
    )

    air_temperature: float = Field(
        ge=295,
        le=305,
        description="Air temperature in Kelvin"
    )

    process_temperature: float = Field(
        ge=305,
        le=315,
        description="Process temperature in Kelvin"
    )

    rotational_speed: float = Field(
        ge=1100,
        le=3000,
        description="Rotational speed in RPM"
    )

    torque: float = Field(
        ge=0,
        le=80,
        description="Torque in Newton-metres"
    )

    tool_wear: float = Field(
        ge=0,
        le=260,
        description="Tool wear in minutes"
    )


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# Metrics endpoint
# --------------------------------------------------

@app.get("/metrics")
def metrics():
    return {
        "total_prediction_requests": request_count
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: MachineInput):
    global request_count

    try:
        # Create the same engineered features used during training
        temperature_difference = (
            data.process_temperature - data.air_temperature
        )

        power = (
            data.torque
            * data.rotational_speed
            * 2
            * np.pi
            / 60
        )

        # Create one row containing raw and engineered features
        row = {
            "Air temperature K": data.air_temperature,
            "Process temperature K": data.process_temperature,
            "Rotational speed rpm": data.rotational_speed,
            "Torque Nm": data.torque,
            "Tool wear min": data.tool_wear,
            "Temperature difference": temperature_difference,
            "Power W": power,
            "Type_H": int(data.machine_type == "H"),
            "Type_L": int(data.machine_type == "L"),
            "Type_M": int(data.machine_type == "M"),
        }

        # Verify that every feature required by the model is available
        missing_columns = [
            column
            for column in feature_columns
            if column not in row
        ]

        if missing_columns:
            raise ValueError(
                f"Missing model features: {missing_columns}"
            )

        # Convert input into a one-row DataFrame
        input_df = pd.DataFrame([row])

        # Match the exact column order used during training
        input_df = input_df[feature_columns]

        # Apply the previously fitted scaler
        input_scaled = scaler.transform(input_df)

        # Generate prediction
        prediction = int(model.predict(input_scaled)[0])

        result = (
            "Machine Failure"
            if prediction == 1
            else "No Machine Failure"
        )

        # Increment only after a successful prediction
        with request_lock:
            request_count += 1

        return {
            "prediction": prediction,
            "result": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed."
        ) from error


# Run using:
# python -m uvicorn main:app --reload