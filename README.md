# Self-Healing Demand Forecaster: Automated Drift Detection & Retraining Engine

An end-to-end **Machine Learning Engineering (MLE)** system that detects covariate and concept drift in production traffic, orchestrates automated retraining via **Prefect**, enforces a **Champion vs. Challenger** quality gate in **MLflow**, and hot-reloads improved models into a **FastAPI** microservice with zero downtime.

---

## System Architecture

```text
  [Production Stream] ──► [FastAPI /predict] ──► [SQLite Telemetry Store]
                                ▲                           │
                                │ (Hot Reload)              ▼
  [MLflow Registry] ◄── [Gatekeeper Eval] ◄── [Prefect + PSI/KS Drift Monitor]
```

## Key Engineering Highlights
* **Zero-Leakage Feature Engineering:** Strict chronological lag/rolling window transformations verified by automated `pytest` assertions (`tests/test_features.py`).
* **Statistical Drift Monitoring:** Tracks Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) p-values across production feature windows (`src/monitor.py`).
* **Automated Deployment Gatekeeper:** Challenger models are only promoted to `@champion` if they reduce holdout MAE by $\ge 2\%$, preventing regressions during benign seasonal fluctuations (`src/evaluate.py`).
* **Production Microservice:** Containerized FastAPI endpoint with Pydantic schema contracts, SQLite telemetry logging, and `/reload-model` hot-swapping.

---

## 3-Month Production Simulation Results

Replaying a 9-month operational timeline (`python -m scripts.simulate_traffic`):

| Simulation Phase | Production Regime | Active Model | Live MAE | System Action |
| :--- | :--- | :--- | :--- | :--- |
| **Month 7** | Normal Summer Load | `v2 (@champion)` | **5.04 MWh** | Seasonal shift triggered retrain (`v3`), **rejected** by gatekeeper (`-8.65%` MAE). |
| **Month 8** | Heatwave + Price Shock | `v2 (@champion)` | **17.23 MWh** | Drift detected (`50%` features); `v4` trained and **promoted** (`+67.53%` MAE gain). |
| **Month 9** | Post-Drift Regime | `v4 (@champion)` | **5.89 MWh** | Hot-reloaded `v4` restored forecast accuracy automatically. |

---

## Quickstart

```bash
# 1. Install dependencies & run CI checks locally
pip install -e ".[dev]"
python -m src.data_generator
python -m src.train --alias champion
pytest -v

# 2. Run the full 3-month drift & self-healing simulation
python -m scripts.simulate_traffic

# 3. Launch containerized MLflow + FastAPI stack
docker-compose up --build
```
