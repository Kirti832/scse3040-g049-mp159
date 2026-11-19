# Group G049 — Problem MP159: Tomorrow's Air Quality (Jaipur)

Course: **SCSE3040 MLOps** (Batch 25)  
Faculty: **Dr. Avantika Saklani**  
Problem: **MP159 — Tomorrow's Air Quality (Jaipur)**  
Group ID: **G049**

## Team Members
- `S24CSEU1269` — AarushiChakraborty
- `S24CSEU1416` — HimanshuKumar
- `S24CSEU0012` — KirtiNarayan

## Repository Layout
- `src/mp159/`: Data audit, data cleaning, training, monitoring/drift, retraining, and inference modules.
- `tests/`: Automated `pytest` suite (including FastAPI `TestClient` tests).
- `data/raw/`: Raw datasets (`train.csv` and `later.csv` tracked via DVC; `holdout_features.csv`).
- `app/main.py`: FastAPI service exposing `GET /health` and `POST /predict` with Pydantic validation and JSON logging.
- `models/model.joblib`: Champion model artifact loaded by the inference service.
- `mlflow.db`: MLflow SQLite tracking and model registry database.
- `Dockerfile` & `.dockerignore`: Container build definition for `mp159-api:1.0`.
- `.github/workflows/ci.yml`: GitHub Actions CI pipeline running `pytest` on push.
- `k8s/deployment.yaml` & `k8s/service.yaml`: Kubernetes manifests (`mp159-api` with 4 replicas, `mp159-svc` on port 80 -> 8000).
- `reports/findings.yaml`: Completed findings and metrics report.
- `predictions/holdout_predictions.csv`: Final predictions (`record_id,prediction`).
