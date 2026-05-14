"""FastAPI inference server for the MAGIC Gamma Telescope classifier.

Loads `artifacts/model_v1.joblib` (the sigmoid-calibrated, Optuna-tuned
XGBoost pipeline) at startup. Threshold and metadata come from
`artifacts/deployment_config.json`.

Endpoints:
    GET  /health        - liveness + model loaded status
    GET  /model_info    - model version, threshold, tuned params
    POST /predict       - single-event classification
    POST /predict_batch - many events at once
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# Config & model loading

ARTIFACT_DIR = Path(os.environ.get("MODEL_DIR", "artifacts"))
MODEL_PATH = ARTIFACT_DIR / "model_v1.joblib"
CONFIG_PATH = ARTIFACT_DIR / "deployment_config.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("magic-gamma-api")

state: dict = {"model": None, "config": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run `python run_full.py` first."
        )
    if not CONFIG_PATH.exists():
        raise RuntimeError(
            f"Deployment config not found at {CONFIG_PATH}. "
            "Run `python run_full.py` first."
        )
    log.info("loading model from %s", MODEL_PATH)
    state["model"] = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH) as f:
        state["config"] = json.load(f)
    log.info(
        "ready. model=%s threshold=%.4f",
        state["config"]["model_type"],
        state["config"]["deployment_threshold"],
    )
    yield
    state.clear()


app = FastAPI(
    title="MAGIC Gamma Telescope classifier",
    description=(
        "Gamma vs hadron classification from 10 Hillas parameters. "
        "Headline model: Optuna-tuned XGBoost with sigmoid calibration. "
        "Threshold chosen at FPR=0.01 on validation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Input schemas

# Pydantic field constraints come from the physical-validity rules in
# src/feature_engineering.py::validate_physical_constraints. Note fWidth
# can be 0 (98 such rows in the original data) — that's why we use ge=0,
# not gt=0.


class HillasInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fLength: float = Field(gt=0, le=500, description="Major axis (mm)")
    fWidth: float = Field(ge=0, le=300, description="Minor axis (mm)")
    fSize: float = Field(ge=0, le=10, description="log10(total photon count)")
    fConc: float = Field(ge=0, le=1, description="Ratio of 2 brightest pixels to total")
    fConc1: float = Field(ge=0, le=1, description="Ratio of brightest pixel to total")
    fAsym: float = Field(description="Shower asymmetry (mm, signed)")
    fM3Long: float = Field(description="3rd moment along major axis (mm)")
    fM3Trans: float = Field(description="3rd moment along minor axis (mm)")
    fAlpha: float = Field(ge=0, le=90, description="Source-pointing angle (deg)")
    fDist: float = Field(gt=0, le=500, description="Distance from camera center (mm)")


class BatchInput(BaseModel):
    events: List[HillasInput] = Field(min_length=1, max_length=10_000)


class PredictionOutput(BaseModel):
    gamma_probability: float
    predicted_class: str  # "g"-gamma or "h"-hadron
    threshold: float


class BatchOutput(BaseModel):
    predictions: List[PredictionOutput]
    n_events: int


# Endpoints


@app.get("/health")
def health():
    return {
        "status": "ok" if state.get("model") is not None else "loading",
        "model_loaded": state.get("model") is not None,
    }


@app.get("/model_info")
def model_info():
    cfg = state.get("config")
    if cfg is None:
        raise HTTPException(503, "model config not loaded")
    return cfg


def _predict_df(events_df: pd.DataFrame) -> np.ndarray:
    model = state.get("model")
    if model is None:
        raise HTTPException(503, "model not loaded")
    # Ensure column order matches what the pipeline expects
    cols = state["config"]["feature_order"]
    events_df = events_df[cols]
    return model.predict_proba(events_df)[:, 1]


@app.post("/predict", response_model=PredictionOutput)
def predict(event: HillasInput):
    df = pd.DataFrame([event.model_dump()])
    p = float(_predict_df(df)[0])
    thr = state["config"]["deployment_threshold"]
    return PredictionOutput(
        gamma_probability=p,
        predicted_class="g" if p >= thr else "h",
        threshold=thr,
    )


@app.post("/predict_batch", response_model=BatchOutput)
def predict_batch(batch: BatchInput):
    df = pd.DataFrame([e.model_dump() for e in batch.events])
    probs = _predict_df(df)
    thr = state["config"]["deployment_threshold"]
    preds = [
        PredictionOutput(
            gamma_probability=float(p),
            predicted_class="g" if p >= thr else "h",
            threshold=thr,
        )
        for p in probs
    ]
    return BatchOutput(predictions=preds, n_events=len(preds))
