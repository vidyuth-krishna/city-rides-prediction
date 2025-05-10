import pandas as pd
import plotly.express as px
from datetime import timedelta
from typing import Optional
import plotly.graph_objects as go

def plot_aggregated_time_series(
    features: pd.DataFrame,
    targets: pd.Series,
    row_id: int,
    predictions: Optional[pd.Series] = None,
):
    location_features = features.iloc[row_id]
    actual_target = targets.iloc[row_id]

    time_series_columns = [col for col in features.columns if col.startswith("rides_t-")]
    time_series_values = [location_features[col] for col in time_series_columns] + [actual_target]

    time_series_dates = pd.date_range(
        start=location_features["pickup_hour"] - timedelta(hours=len(time_series_columns)),
        end=location_features["pickup_hour"],
        freq="h",
    )

    title = f"Pickup Hour: {location_features['pickup_hour']}, Location ID: {location_features['pickup_location_id']}"

    fig = px.line(
        x=time_series_dates,
        y=time_series_values,
        template="plotly_white",
        markers=True,
        title=title,
        labels={"x": "Time", "y": "Ride Counts"},
    )

    fig.add_scatter(
        x=time_series_dates[-1:],
        y=[actual_target],
        mode="markers",
        marker=dict(color="green", size=10),
        name="Actual Value",
    )

    if predictions is not None:
        predicted_value = predictions.loc[predictions.index == row_id]
        fig.add_scatter(
            x=time_series_dates[-1:],
            y=predicted_value.values,
            mode="markers",
            marker=dict(color="red", symbol="x", size=12),
            name="Prediction",
        )

    return fig

def plot_prediction(features: pd.DataFrame, prediction: pd.DataFrame):
    time_series_columns = [col for col in features.columns if col.startswith("rides_t-")]
    time_series_values = [features[col].iloc[0] for col in time_series_columns] + prediction["predicted_demand"].to_list()

    pickup_hour = pd.Timestamp(features["pickup_hour"].iloc[0])

    time_series_dates = pd.date_range(
        start=pickup_hour - timedelta(hours=len(time_series_columns)),
        end=pickup_hour,
        freq="h",
    )

    historical_df = pd.DataFrame({
        "datetime": time_series_dates,
        "rides": time_series_values
    })

    title = f"Pickup Hour: {pickup_hour}, Location ID: {features['pickup_location_id'].iloc[0]}"

    fig = px.line(
        historical_df,
        x="datetime",
        y="rides",
        template="plotly_white",
        markers=True,
        title=title,
        labels={"datetime": "Time", "rides": "Ride Counts"},
    )

    fig.add_scatter(
        x=[pickup_hour],
        y=prediction["predicted_demand"].to_list(),
        line_color="red",
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        name="Prediction",
    )

    return fig
