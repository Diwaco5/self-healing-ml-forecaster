from contextlib import asynccontextmanager
from typing import Any

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.api.schemas import PredictionRequest, PredictionResponse
from src.config import settings
from src.features import FEATURE_COLUMNS

engine = create_engine(settings.DB_URI, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, index=True)
    ambient_temp = Column(Float)
    industrial_activity = Column(Float)
    grid_price_index = Column(Float)
    is_weekend = Column(Integer)
    hour = Column(Integer)
    demand_lag_1h = Column(Float)
    demand_lag_24h = Column(Float)
    demand_rolling_mean_24h = Column(Float)
    predicted_demand_mwh = Column(Float)
    actual_demand_mwh = Column(Float, nullable=True)
    model_version = Column(String)


Base.metadata.create_all(bind=engine)

model_state: dict[str, Any] = {"model": None, "version": "unloaded"}


def load_champion_model() -> None:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
    meta = client.get_model_version_by_alias(settings.MODEL_NAME, "champion")
    model_state["model"] = mlflow.pyfunc.load_model(f"models:/{settings.MODEL_NAME}@champion")
    model_state["version"] = f"v{meta.version}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_champion_model()
    except Exception as exc:
        print(f"Warning: Could not load @champion on startup: {exc}")
    yield


app = FastAPI(title="Self-Healing Demand Forecaster API", lifespan=lifespan)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "model_version": model_state["version"]}


@app.post("/reload-model")
def reload_model() -> dict[str, str]:
    load_champion_model()
    return {"status": "reloaded", "active_version": model_state["version"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Champion model not loaded.")

    input_df = pd.DataFrame([payload.model_dump()])[FEATURE_COLUMNS].astype("float64")
    preds = model_state["model"].predict(input_df)
    prediction = float(preds[0])

    with SessionLocal() as session:
        log_entry = PredictionLog(
            timestamp=payload.timestamp,
            ambient_temp=payload.ambient_temp,
            industrial_activity=payload.industrial_activity,
            grid_price_index=payload.grid_price_index,
            is_weekend=payload.is_weekend,
            hour=payload.hour,
            demand_lag_1h=payload.demand_lag_1h,
            demand_lag_24h=payload.demand_lag_24h,
            demand_rolling_mean_24h=payload.demand_rolling_mean_24h,
            predicted_demand_mwh=round(prediction, 2),
            actual_demand_mwh=payload.actual_demand_mwh,
            model_version=model_state["version"],
        )
        session.add(log_entry)
        session.commit()

    return PredictionResponse(
        predicted_demand_mwh=round(prediction, 2),
        model_version=model_state["version"],
    )
