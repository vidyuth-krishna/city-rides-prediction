import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pathlib import Path

# === Streamlit UI setup ===
st.set_page_config(layout="wide")
st.title("📊 Citi Bike Model Monitoring")
st.markdown("Track Mean Absolute Error (MAE) over time")

# Sidebar slider to choose the number of past hours to monitor
st.sidebar.header("Monitoring Settings")
past_hours = st.sidebar.slider(
    "Past Hours to Plot", min_value=12, max_value=24*28, value=24*7, step=12
)

# === Load prediction data ===
results_path = Path("data/transformed/predictions_with_targets.parquet")
if not results_path.exists():
    st.error("❌ Prediction results file not found. Run the inference pipeline first.")
    st.stop()

df = pd.read_parquet(results_path)

# Standardize column names
if "start_hour" in df.columns:
    df = df.rename(columns={"start_hour": "pickup_hour"})

# Ensure datetime and local timezone conversion
df["pickup_hour"] = pd.to_datetime(df["pickup_hour"], utc=True).dt.tz_convert("America/New_York")

# Filter to last N hours
latest_time = df["pickup_hour"].max()
cutoff_time = latest_time - pd.Timedelta(hours=past_hours)
recent_df = df[df["pickup_hour"] > cutoff_time]

if recent_df.empty:
    st.warning("⚠️ Not enough data after cutoff. Try increasing the window.")
    st.stop()

# Compute absolute error
if "predicted" not in recent_df.columns or "target" not in recent_df.columns:
    st.error("❌ 'predicted' or 'target' column is missing.")
    st.stop()

recent_df["absolute_error"] = abs(recent_df["predicted"] - recent_df["target"])

# Group by time and calculate MAE
mae_by_hour = (
    recent_df.groupby("pickup_hour")["absolute_error"]
    .mean()
    .reset_index()
    .rename(columns={"absolute_error": "MAE"})
)

# Plot the trend
fig = px.line(
    mae_by_hour,
    x="pickup_hour",
    y="MAE",
    title=f"Mean Absolute Error (MAE) for Past {past_hours} Hours",
    labels={"pickup_hour": "Hour", "MAE": "Mean Absolute Error"},
    markers=True,
)

# Show results
st.plotly_chart(fig, use_container_width=True)
st.metric("📉 Average MAE", f"{mae_by_hour['MAE'].mean():.2f}")
