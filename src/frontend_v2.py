import sys
from pathlib import Path

# Add project root to sys.path so `src/` can be imported
sys.path.append(str(Path("..").resolve()))

import pandas as pd
import numpy as np
import joblib
import streamlit as st
from datetime import datetime
from src.plot_utils import plot_aggregated_time_series
from src.data_utils import transform_ts_data_info_features_and_target_loop

# === 1. Load data ===
data_path = Path("../data/transformed/top_3_hourly_timeseries.parquet")
df = pd.read_parquet(data_path)

# Convert to timezone-aware (UTC), then to local time
df["start_hour"] = pd.to_datetime(df["start_hour"], utc=True).dt.tz_convert("America/New_York")

# === 2. Load model ===
model_path = Path("models/lgbm_model_28day.pkl")
model = joblib.load(model_path)

# === 3. Transform to supervised format ===
features_df, targets = transform_ts_data_info_features_and_target_loop(
    df,
    window_size=24 * 28,
    step_size=24
)

# === 4. Add predictions ===
non_feature_cols = ["start_hour", "start_station_id", "date"]
input_data = features_df.drop(columns=non_feature_cols, errors="ignore")
preds = np.round(model.predict(input_data)).astype(int)
features_df["predicted_demand"] = preds
features_df["target"] = targets.astype(int)

# === 5. Rename cols for plotting and apply timezone again ===
features_df = features_df.rename(columns={
    "start_hour": "pickup_hour",
    "start_station_id": "pickup_location_id"
})
features_df["pickup_hour"] = pd.to_datetime(features_df["pickup_hour"], utc=True).dt.tz_convert("America/New_York")

# === 6. UI Setup ===
st.set_page_config(layout="wide")
local_now = datetime.now().astimezone().strftime("%Y-%m-%d %I:%M %p %Z")
st.title("🚲 Citi Bike Demand Prediction")
st.markdown(f"**As of {local_now}** — Predicting next-hour demand for top 3 NYC Citi Bike stations.")

station_map = {
    6140: "W 21 St & 6 Ave",
    6822: "1 Ave & E 68 St",
    5905: "University Pl & E 14 St"
}
selected_id = st.selectbox("Select a station", options=list(station_map.keys()), format_func=lambda x: station_map[x])

# === 7. Filter and plot ===
filtered_df = features_df[features_df["pickup_location_id"] == selected_id]
latest_row = filtered_df.sort_values("pickup_hour").iloc[-1]
row_id = latest_row.name

fig = plot_aggregated_time_series(
    features=features_df,
    targets=features_df["target"],
    row_id=row_id,
    predictions=features_df["predicted_demand"]
)

st.plotly_chart(fig, use_container_width=True)
