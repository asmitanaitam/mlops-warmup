import json
import os

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
NAME = "flood-risk-model"
ALIAS = "champion"
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

# 1. Ask the registry which version is the champion
client = MlflowClient()
mv = client.get_model_version_by_alias(NAME, ALIAS)
run = client.get_run(mv.run_id)
print(f"Champion is version {mv.version} from run {mv.run_id}")

# 2. Load that model from the registry
uri = f"models:/{NAME}@{ALIAS}"
try:
    model = mlflow.sklearn.load_model(uri, skops_trusted_types=TRUSTED_TYPES)
except TypeError:
    model = mlflow.sklearn.load_model(uri)

# 3. Rebuild the same training data to record what the model has seen
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
X_train, _, _, _ = train_test_split(
    data[features], data["risk"], test_size=0.2, random_state=42
)

reference = {
    f: {
        "min": float(X_train[f].min()),
        "max": float(X_train[f].max()),
        "mean": float(X_train[f].mean()),
        "std": float(X_train[f].std()),
    }
    for f in features
}

# 4. Save everything the API needs
joblib.dump({"model": model, "features": features}, "model.joblib")
with open("reference_stats.json", "w") as f:
    json.dump(reference, f, indent=2)
with open("model_info.json", "w") as f:
    json.dump(
        {
            "name": NAME,
            "version": str(mv.version),
            "run_id": mv.run_id,
            "mae": run.data.metrics.get("mae"),
        },
        f,
        indent=2,
    )
print("Saved model.joblib, reference_stats.json, model_info.json")