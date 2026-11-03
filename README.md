\# MLOps Warm-up: Flood Risk Model Service



A small end-to-end MLOps project: train a model, track experiments, version the best model, serve it through an API in a Docker container, and monitor incoming data for drift.



The model uses synthetic data on purpose. The goal is the pipeline around the model, not the model itself. The same pattern is applied to a real model in my `floodrisk` package.



\## Problem



A trained model on a laptop is not useful to anyone else. This project answers: how do you turn a model into a reproducible, versioned, monitored service?



\## Data



Synthetic data for 2,000 locations with three features (rainfall in mm, elevation in m, slope in degrees) and a flood risk score between 0 and 1. Risk rises with rainfall and falls with elevation and slope, plus random noise. Synthetic data lets me know the true pattern and the noise floor.



\## Approach



```mermaid

flowchart LR

&#x20;   A\[train\_mlflow.py<br/>4 Random Forest runs] --> B\[(MLflow tracking<br/>params, metrics, models)]

&#x20;   B --> C\[Model registry<br/>flood-risk-model, alias: champion]

&#x20;   C --> D\[export\_champion.py<br/>model + training stats]

&#x20;   D --> E\[Docker image<br/>FastAPI service]

&#x20;   E --> F\[POST /predict<br/>validation + logging]

&#x20;   E --> G\[GET /drift<br/>drift check]

&#x20;   F --> H\[(predictions.jsonl<br/>Docker volume)]

&#x20;   H --> G

```



| Piece | Tool | Why |

|---|---|---|

| Experiment tracking | MLflow | Every run's settings, metrics and model are recorded and comparable |

| Model registry | MLflow | The best model gets a version and a `champion` alias, so deployment does not depend on file paths |

| Serving | FastAPI | Turns the model into an API with input validation and auto-generated docs |

| Packaging | Docker | Same environment everywhere, with pinned library versions |

| Monitoring | Prediction log + drift check | Detects inputs that look unlike the training data |

| Persistence | Docker volume | Logs survive container restarts |



\## Results



\*\*Model comparison (test set, 20% held out):\*\*



| Settings | MAE | R2 |

|---|---|---|

| 50 trees, max depth 3 | 0.10 | 0.85 |

| 100 trees, max depth 6 | 0.05 | 0.96 |

| 200 trees, max depth 10 (champion) | 0.04 | 0.97 |

| 200 trees, no depth limit | 0.04 | 0.97 |



Going from depth 3 to depth 10 cut MAE from 0.10 to 0.04. Beyond that the error stopped improving because it reached the noise floor built into the data.



\*\*Drift detection (40 requests each):\*\*



| Traffic | Status | Out-of-range rate | Mean shift (training std) |

|---|---|---|---|

| Normal | ok | 0.0 | 0.02, 0.07, 0.10 |

| Unusual | DRIFT DETECTED | 1.0 | 9.11, -2.28, 10.99 |



\*\*Persistence test:\*\* after 80 logged requests, the container was stopped and restarted. `/drift` still reported 80 requests, so the volume kept the logs.



\## How to run



Requires Python 3.12 and Docker.



```bash

git clone https://github.com/asmitanaitam/mlops-warmup.git

cd mlops-warmup

python -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements.txt mlflow



\# 1. Train 4 models and register the best as "champion"

python train\_mlflow.py



\# 2. Export the champion and training statistics

python export\_champion.py



\# 3. Build and run the service (Windows cmd syntax for the volume)

docker build -t flood-risk-api:v2 .

mkdir docker\_logs

docker run -p 8000:8000 -v "%cd%\\docker\_logs:/app/logs" flood-risk-api:v2



\# 4. Send test traffic (in a second terminal) and check drift

python send\_traffic.py normal

python send\_traffic.py unusual

```



Open http://127.0.0.1:8000/docs for the interactive API page. To browse experiments: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.



Generated files (`model.joblib`, `mlflow.db`, `mlruns/`, logs) are not in the repo. Steps 1 and 2 recreate them.



\## Challenges and solutions



| Challenge | Cause | Solution |

|---|---|---|

| `python` not found in PowerShell | Anaconda does not add Python to PATH | Used Anaconda Prompt and a dedicated conda environment |

| MLflow crashed while saving the model | MLflow 3 saves scikit-learn models in skops format, which rejects unknown types such as `sklearn.tree.\_tree.Tree` | Whitelisted only that one type, since I trained the model myself |

| Container crashed on startup with FileNotFoundError | The Dockerfile edit was never saved, so new files were not copied into the image | Verified the file with `type Dockerfile` before rebuilding |

| Model gave confident answers for impossible inputs | Validation only blocks invalid values, and Random Forests cannot extrapolate | Added training-range checks and a drift endpoint |

| Logs lost on container restart | A container's filesystem is temporary | Mounted a volume and verified by counting log lines before and after a restart |



\## Limitations



\- Synthetic data, so the metrics say nothing about real flood risk.

\- The drift rule is simple (mean shift and out-of-range share). Production systems often use PSI or Kolmogorov-Smirnov tests.

\- Drift is checked on demand, not on a schedule, and there are no alerts.

\- No authentication on the API.



\## Next



The same pipeline applied to `floodrisk`, a LightGBM flood risk model with spatial cross-validation.

