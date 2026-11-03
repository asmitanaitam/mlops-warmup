import os

# Silence the harmless "Git not found" warnings (must be set before importing mlflow)
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Store runs in a local database file (needed for the model registry)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("flood-risk-warmup")

# Types we explicitly trust when saving the model (we created the model ourselves)
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

# 1. Same fake data as before
rng = np.random.default_rng(42)
n = 2000
data = pd.DataFrame({
    "rainfall_mm": rng.gamma(2.0, 30.0, n),
    "elevation_m": rng.uniform(0, 300, n),
    "slope_deg": rng.uniform(0, 15, n),
})
score = (
    0.03 * data["rainfall_mm"]
    - 0.02 * data["elevation_m"]
    - 0.10 * data["slope_deg"]
    + 1.5
    + rng.normal(0, 0.3, n)
)
data["risk"] = 1 / (1 + np.exp(-score))

features = ["rainfall_mm", "elevation_m", "slope_deg"]
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data["risk"], test_size=0.2, random_state=42
)

# 2. Settings to compare. Each one becomes one MLflow run.
configs = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 6},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 200, "max_depth": None},
]

best = {"mae": float("inf"), "run_id": None}

for cfg in configs:
    with mlflow.start_run() as run:
        model = RandomForestRegressor(random_state=42, **cfg)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)

        # Record settings, results, and the model file itself
        mlflow.log_params({k: str(v) for k, v in cfg.items()})
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train.head(3),
            skops_trusted_types=TRUSTED_TYPES,
        )

        print(f"{cfg} -> MAE {mae:.4f}  RMSE {rmse:.4f}  R2 {r2:.4f}")

        if mae < best["mae"]:
            best = {"mae": mae, "run_id": run.info.run_id}

# 3. Put the best run's model on the registry shelf
print(f"\nBest run: {best['run_id']} with MAE {best['mae']:.4f}")
result = mlflow.register_model(f"runs:/{best['run_id']}/model", "flood-risk-model")

# Label this version so we can ask for it by name later
client = MlflowClient()
client.set_registered_model_alias("flood-risk-model", "champion", result.version)
print(f"Registered flood-risk-model version {result.version} with alias 'champion'")