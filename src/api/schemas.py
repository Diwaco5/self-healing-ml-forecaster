from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    timestamp: str = Field(..., examples=["2026-07-01 12:00:00"])
    ambient_temp: float = Field(..., ge=-50.0, le=150.0)
    industrial_activity: float = Field(..., ge=0.0, le=200.0)
    grid_price_index: float = Field(..., ge=0.0)
    is_weekend: int = Field(..., ge=0, le=1)
    hour: int = Field(..., ge=0, le=23)
    demand_lag_1h: float = Field(..., ge=0.0)
    demand_lag_24h: float = Field(..., ge=0.0)
    demand_rolling_mean_24h: float = Field(..., ge=0.0)
    actual_demand_mwh: float | None = Field(default=None, description="Optional delayed ground truth")


class PredictionResponse(BaseModel):
    predicted_demand_mwh: float
    model_version: str
