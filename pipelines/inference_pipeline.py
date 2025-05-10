import pandas as pd
import joblib
import hopsworks
from datetime import datetime, timedelta
from pathlib import Path

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    FEATURE_GROUP_MODEL_PREDICTION,
    MODELS_DIR
)
from src.data_utils import transform_ts_data_info_features

# === 1. Connect to Hopsworks ===
project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project=HOPSWORKS_PROJECT_NAME
)
fs = project.get_feature_store()

# === 2. Time range to fetch ===
predict_for_hour = pd.Timestamp.now(tz="UTC").ceil("h")
start_time = predict_for_hour - timedelta(days=28)
end_time = predict_for_hour - timedelta(hours=1)

print(f"⏳ Fetching data from {start_time} to {end_time}")
fv = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
ts_data = fv.get_batch_data(start_time=start_time, end_time=end_time)

ts_data = ts_data[ts_data["start_hour"].between(start_time, end_time)]
ts_data = ts_data.sort_values(["start_station_id", "start_hour"]).reset_index(drop=True)

# === 3. Create lag-based features ===
ts_data["start_hour"] = pd.to_datetime(ts_data["start_hour"])
features = transform_ts_data_info_features(
    df=ts_data,
    window_size=24 * 28,
    step_size=23
)

if features.empty:
    raise ValueError("❌ Not enough data to generate features for inference.")

# === 4. Load model from disk (assumes you've run training pipeline already) ===
model_path = Path(MODELS_DIR) / "lgbm_model_28day.pkl"
model = joblib.load(model_path)

# === 5. Predict ===
X = features.drop(columns=["pickup_hour", "pickup_location_id"])
preds = model.predict(X)
preds = preds.round().astype(int)

predictions_df = features[["pickup_hour", "pickup_location_id"]].copy()
predictions_df["predicted_demand"] = preds
predictions_df.rename(columns={
    "pickup_hour": "start_hour",
    "pickup_location_id": "start_station_id"
}, inplace=True)
predictions_df["start_hour"] = predict_for_hour  # override just in case

print("✅ Predictions:")
print(predictions_df.head())

# === 6. Insert predictions to Hopsworks ===
prediction_fg = fs.get_or_create_feature_group(
    name=FEATURE_GROUP_MODEL_PREDICTION,
    version=1,
    description="LGBM hourly predictions for top 3 Citi Bike stations",
    primary_key=["start_station_id", "start_hour"],
    event_time="start_hour",
)

prediction_fg.insert(predictions_df, write_options={"wait_for_job": False})
print("✅ Predictions inserted into Hopsworks!")
