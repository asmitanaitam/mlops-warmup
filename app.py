import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

LOG_PATH = os.environ.get("LOG_PATH", "logs/predictions.jsonl")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = joblib.load("model.joblib")
    state["model"] = artifact["model"]
    state["features"] = artifact["features"]
    with open("reference_stats.json") as f:
        state["reference"] = json.load(f)
    with open("model_info.json") as f:
        state["info"] = json.load(f)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    yield
    state.clear()


app = FastAPI(title="Flood Risk API (warm-up)", lifespan=lifespan)


class RiskRequest(BaseModel):
    rainfall_mm: float = Field(..., ge=0, le=1000)
    elevation_m: float = Field(..., ge=-100, le=9000)
    slope_deg: float = Field(..., ge=0, le=90)


class RiskResponse(BaseModel):
    risk: float
    model_version: str
    in_training_range: bool
    out_of_range_features: list[str]


def out_of_range(inputs: dict) -> list[str]:
    ref = state["reference"]
    return [
        f for f in state["features"]
        if not (ref[f]["min"] <= inputs[f] <= ref[f]["max"])
    ]


@app.get("/health")
def health():
    return {"status": "ok", "model": state["info"]}


@app.post("/predict", response_model=RiskResponse)
def predict(request: RiskRequest):
    inputs = request.model_dump()
    row = pd.DataFrame([inputs])[state["features"]]
    risk = float(state["model"].predict(row)[0])
    flagged = out_of_range(inputs)

    # Prediction logging: one JSON line per request
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "risk": risk,
        "model_version": state["info"]["version"],
        "out_of_range_features": flagged,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    return RiskResponse(
        risk=risk,
        model_version=state["info"]["version"],
        in_training_range=not flagged,
        out_of_range_features=flagged,
    )


@app.get("/drift")
def drift(last_n: int = 200):
    if not os.path.exists(LOG_PATH):
        return {"status": "no data yet"}
    with open(LOG_PATH) as f:
        lines = f.readlines()[-last_n:]
    records = [json.loads(line) for line in lines]
    if len(records) < 20:
        return {"status": "not enough data", "n_requests": len(records), "needed": 20}

    recent = pd.DataFrame([r["inputs"] for r in records])
    ref = state["reference"]
    report = {}
    shifted = False
    for feat in state["features"]:
        # How far the recent average moved, in units of training spread
        shift = (recent[feat].mean() - ref[feat]["mean"]) / ref[feat]["std"]
        report[feat] = {
            "recent_mean": round(float(recent[feat].mean()), 2),
            "training_mean": round(ref[feat]["mean"], 2),
            "shift_in_std": round(float(shift), 2),
        }
        if abs(shift) > 0.5:
            shifted = True

    oor_rate = sum(1 for r in records if r["out_of_range_features"]) / len(records)
    return {
        "status": "DRIFT DETECTED" if (shifted or oor_rate > 0.10) else "ok",
        "n_requests": len(records),
        "out_of_range_rate": round(oor_rate, 3),
        "features": report,
        "rule": "drift if any mean shifts > 0.5 training std, or > 10% of requests are out of training range",
    }