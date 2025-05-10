import numpy as np
import pandas as pd

def transform_ts_data_info_features_and_target_loop(
    df,
    feature_col="rides",
    location_col="start_station_id",
    time_col="start_hour",
    window_size=24 * 28,
    step_size=24,
):
    """
    Transforms hourly time series data for each unique location into tabular format using a sliding window.

    Args:
        df (pd.DataFrame): DataFrame with time series data (hourly resolution).
        feature_col (str): Column to use as input features and target (default: "rides").
        location_col (str): Column identifying location (default: "start_station_id").
        time_col (str): Column with timestamp info (default: "start_hour").
        window_size (int): Number of past hours to use as features.
        step_size (int): Step size to slide window.

    Returns:
        features (pd.DataFrame), targets (pd.Series)
    """
    location_ids = df[location_col].unique()
    transformed_data = []

    for location_id in location_ids:
        location_data = df[df[location_col] == location_id].reset_index(drop=True)
        values = location_data[feature_col].values
        times = location_data[time_col].values

        if len(values) <= window_size:
            print(f"⚠️ Skipping station {location_id} (not enough data)")
            continue

        rows = []
        for i in range(0, len(values) - window_size, step_size):
            features = values[i : i + window_size]
            target = values[i + window_size]
            target_time = times[i + window_size]
            row = np.append(features, [target, location_id, target_time])
            rows.append(row)

        feature_columns = [f"{feature_col}_t-{window_size - i}" for i in range(window_size)]
        all_columns = feature_columns + ["target", location_col, time_col]
        transformed_df = pd.DataFrame(rows, columns=all_columns)
        transformed_data.append(transformed_df)

    if not transformed_data:
        raise ValueError("❌ No location had enough data for transformation.")

    final_df = pd.concat(transformed_data, ignore_index=True)
    features = final_df[feature_columns + [time_col, location_col]]
    targets = final_df["target"]
    return features, targets

from typing import Tuple
import pandas as pd
from datetime import datetime

def split_time_series_data(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits a time series DataFrame into training and testing sets based on a cutoff date.

    Args:
        df (pd.DataFrame): The input DataFrame containing the time series data.
        cutoff_date (datetime): The date used to split the data into training and testing sets.
        target_column (str): The name of the target column to separate from the features.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            - X_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training target values.
            - X_test (pd.DataFrame): Testing features.
            - y_test (pd.Series): Testing target values.
    """
    train_data = df[df["pickup_hour"] < cutoff_date].reset_index(drop=True)
    test_data = df[df["pickup_hour"] >= cutoff_date].reset_index(drop=True)

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    return X_train, y_train, X_test, y_test

def transform_ts_data_info_features_and_target(
    df,
    feature_col="rides",
    window_size=24*28,  # 28 days
    step_size=24        # predict every next day
):
    """
    Transforms time series data for all unique location IDs into a supervised learning format.
    Features = previous `window_size` hours of rides.
    Target = next hour ride count.

    Parameters:
        df (pd.DataFrame): Must include 'pickup_hour', 'pickup_location_id', and the feature_col.
        feature_col (str): Column name for the target time series (default = "rides")
        window_size (int): Number of lag hours to use as features
        step_size (int): Sliding window step size

    Returns:
        features (pd.DataFrame), targets (pd.Series)
    """

    location_ids = df["pickup_location_id"].unique()
    transformed_data = []

    for location_id in location_ids:
        location_data = df[df["pickup_location_id"] == location_id].reset_index(drop=True)

        values = location_data[feature_col].values
        times = location_data["pickup_hour"].values

        if len(values) <= window_size:
            continue

        for i in range(0, len(values) - window_size, step_size):
            features = values[i : i + window_size]
            target = values[i + window_size]
            target_time = times[i + window_size]

            row = np.append(features, [target, location_id, target_time])
            transformed_data.append(row)

    feature_columns = [f"{feature_col}_t-{window_size - i}" for i in range(window_size)]
    all_columns = feature_columns + ["target", "pickup_location_id", "pickup_hour"]

    final_df = pd.DataFrame(transformed_data, columns=all_columns)

    features = final_df[feature_columns + ["pickup_hour", "pickup_location_id"]]
    targets = final_df["target"]

    return features, targets