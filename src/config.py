from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MLRUNS_DIR: Path = BASE_DIR / "mlruns"
    REFERENCE_DATA_PATH: Path = DATA_DIR / "reference_train.parquet"
    STREAM_DATA_PATH: Path = DATA_DIR / "production_stream.parquet"
    DB_URI: str = f"sqlite:///{DATA_DIR / 'production_logs.db'}"

    # Uses local SQLite tracking store out of the box; override in Docker Compose
    MLFLOW_TRACKING_URI: str = f"sqlite:///{DATA_DIR / 'mlflow.db'}"
    EXPERIMENT_NAME: str = "energy-demand-forecaster"
    MODEL_NAME: str = "demand-forecaster"

    DRIFT_SHARE_THRESHOLD: float = 0.30
    MIN_IMPROVEMENT_PCT: float = 0.02
    RANDOM_SEED: int = 42


settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
