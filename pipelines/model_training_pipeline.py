import pandas as pd
import numpy as np
import hopsworks
import joblib
from lightgbm import LGBMRegressor
from datetime import datetime, timedelta
from pathlib import Path

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    MODELS_DIR
)
from src.data_utils import (
    transform_ts_data_info_features_and_target,
    split_time_series_data
)
from src.experiment_utils import set_mlflow_tracking, log_model_to_mlflow

# === 1. Connect to Hopsworks ===
project = hopsworks.login(
    project=HOPSWORKS_PROJECT_NAME,
    api_key_value=HOPSWORKS_API_KEY
)
fs = project.get_feature_store()
fv = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)

# === 2. Load full year of data (2023-05 to 2024-04) ===
ts_data, _ = fv.training_data(
    start_time=datetime(2023, 5, 1),
    end_time=datetime(2024, 4, 30)
)
print(f"✅ Loaded shape: {ts_data.shape}")
print(ts_data.head())
print(ts_data["start_hour"].min(), ts_data["start_hour"].max())

ts_data = ts_data.sort_values(["start_station_id", "start_hour"]).reset_index(drop=True)

# === 3. Transform to tabular features/targets ===
features_df, targets = transform_ts_data_info_features_and_target(
    ts_data,
    window_size=24 * 28,  # 28 days
    step_size=24          # predict next day
)
features_df["target"] = pd.to_numeric(targets, errors="coerce")
features_df["start_hour"] = pd.to_datetime(features_df["start_hour"])

# === 4. Split into train/test ===
cutoff = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
print(f"📆 Using cutoff: {cutoff}")
X_train, y_train, X_test, y_test = split_time_series_data(
    features_df, cutoff_date=cutoff, target_column="target"
)

if X_train.empty:
    raise ValueError("❌ X_train is empty — likely not enough training rows before cutoff.")
if X_test.empty:
    raise ValueError("❌ X_test is empty — check if test range is within your dataset.")

# === 5. Clean feature dtypes ===
lag_cols = [col for col in X_train.columns if col.startswith("rides_t-")]
X_train[lag_cols] = X_train[lag_cols].apply(pd.to_numeric, errors="coerce")
X_test[lag_cols] = X_test[lag_cols].apply(pd.to_numeric, errors="coerce")
y_train = pd.to_numeric(y_train, errors="coerce")
y_test = pd.to_numeric(y_test, errors="coerce")

# === 6. Train model ===
model = LGBMRegressor(random_state=42)
model.fit(X_train.drop(columns=["start_hour", "start_station_id"]), y_train)
train_preds = np.round(model.predict(X_train.drop(columns=["start_hour", "start_station_id"]))).astype(int)
test_preds = np.round(model.predict(X_test.drop(columns=["start_hour", "start_station_id"]))).astype(int)

train_mae = np.round(np.mean(np.abs(train_preds - y_train)), 4)
test_mae = np.round(np.mean(np.abs(test_preds - y_test)), 4)

print(f"✅ Trained LGBM — Train MAE: {train_mae}, Test MAE: {test_mae}")

# === 7. Save model ===
Path(MODELS_DIR).mkdir(exist_ok=True, parents=True)
model_path = Path(MODELS_DIR) / "lgbm_model_28day.pkl"
joblib.dump(model, model_path)

# === 8. Log to MLflow ===
mlflow = set_mlflow_tracking()
log_model_to_mlflow(
    model=model,
    input_data=X_test.drop(columns=["start_hour", "start_station_id"]),
    experiment_name="LGBMModel28Day",
    metric_name="mae",
    score=test_mae,
    model_name="citibike_hourly_predictor"
)

print("✅ Model training pipeline complete.")
