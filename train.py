import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. Make fake but sensible data: more rain and lower, flatter land means higher risk
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
data["risk"] = 1 / (1 + np.exp(-score))  # squeeze into 0 to 1

# 2. Split into training and test sets
features = ["rainfall_mm", "elevation_m", "slope_deg"]
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data["risk"], test_size=0.2, random_state=42
)

# 3. Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Check quality on data the model has not seen
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"Test MAE: {mae:.4f}")

# 5. Save the trained model and its feature order to disk
joblib.dump({"model": model, "features": features}, "model.joblib")
print("Saved model.joblib")