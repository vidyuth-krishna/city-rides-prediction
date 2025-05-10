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

# === 2. Fetch 180 days of data ===
ts_data, _ = fv.training_data(start_time=datetime.now() - timedelta(days=180))
ts_data = ts_data.sort_values(["start_station_id", "start_hour"]).reset_index(drop=True)


# === 3. Transform to tabular features/targets ===
features_df, targets = transform_ts_data_info_features_and_target(
    ts_data,
    window_size=24*28,
    step_size=24
)
features_df["target"] = pd.to_numeric(targets, errors="coerce")
features_df["pickup_hour"] = pd.to_datetime(features_df["pickup_hour"])

# === 4. Split train/test ===
cutoff = pd.Timestamp(datetime.now(), tz="UTC") - timedelta(days=28)
X_train, y_train, X_test, y_test = split_time_series_data(
    features_df, cutoff_date=cutoff, target_column="target"
)

# === 5. Clean feature dtypes ===
lag_cols = [col for col in X_train.columns if col.startswith("rides_t-")]
X_train[lag_cols] = X_train[lag_cols].apply(pd.to_numeric, errors="coerce")
y_train = pd.to_numeric(y_train, errors="coerce")

# === 6. Train model ===
model = LGBMRegressor(random_state=42)
model.fit(X_train.drop(columns=["pickup_hour", "pickup_location_id"]), y_train)
train_preds = np.round(model.predict(X_train.drop(columns=["pickup_hour", "pickup_location_id"]))).astype(int)
train_mae = np.round(np.mean(np.abs(train_preds - y_train)), 4)

print(f"✅ Trained LGBM — MAE: {train_mae}")

# === 7. Save model locally ===
Path(MODELS_DIR).mkdir(exist_ok=True, parents=True)
model_path = Path(MODELS_DIR) / "lgbm_model_28day.pkl"
joblib.dump(model, model_path)

# === 8. Log to MLflow ===
mlflow = set_mlflow_tracking()
log_model_to_mlflow(
    model=model,
    input_data=X_train.drop(columns=["pickup_hour", "pickup_location_id"]),
    experiment_name="LGBMModel28Day",
    metric_name="mae",
    score=train_mae,
    model_name="citibike_hourly_predictor"
)

print("✅ Model training pipeline complete.")
