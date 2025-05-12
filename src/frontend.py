import sys
from pathlib import Path

# Add project root to sys.path so `src/` can be imported
sys.path.append(str(Path("..").resolve()))

import pandas as pd
import numpy as np
import joblib
import streamlit as st
from pathlib import Path
from datetime import datetime
from src.plot_utils import plot_aggregated_time_series
from src.data_utils import transform_ts_data_info_features_and_target_loop

# === 1. Load data ===
data_path = Path("../data/transformed/top_3_hourly_timeseries.parquet")
df = pd.read_parquet(data_path)

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
non_feature_cols = ["start_hour", "start_station_id", "date"]  # drop non-numeric cols
input_data = features_df.drop(columns=non_feature_cols, errors="ignore")
preds = np.round(model.predict(input_data)).astype(int)
features_df["predicted_demand"] = preds
features_df["target"] = targets.astype(int)

# === 5. UI Setup ===
st.set_page_config(layout="wide")
st.title("🚲 Citi Bike Demand Prediction")
st.markdown("Predicting next-hour ride demand for top 3 NYC Citi Bike stations")

# Dropdown to pick station
station_map = {
    6140: "W 21 St & 6 Ave",
    6822: "1 Ave & E 68 St",
    5905: "University Pl & E 14 St"
}
selected_id = st.selectbox("Select a station", options=list(station_map.keys()), format_func=lambda x: station_map[x])

# Filter and plot
filtered_df = features_df[features_df["start_station_id"] == selected_id]
latest_row = filtered_df.sort_values("start_hour").iloc[-1]
row_id = latest_row.name

fig = plot_aggregated_time_series(
    features=features_df.rename(columns={
        "start_hour": "pickup_hour",
        "start_station_id": "pickup_location_id"
    }),
    targets=features_df["target"],
    row_id=row_id,
    predictions=features_df["predicted_demand"]
)



st.plotly_chart(fig, use_container_width=True)
