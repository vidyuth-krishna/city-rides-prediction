import pandas as pd
from datetime import datetime
import hopsworks
from src.config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
from pathlib import Path

def load_hourly_timeseries_data():
    path = Path("data/transformed/top_3_hourly_timeseries.parquet")
    df = pd.read_parquet(path)
    df["start_hour"] = pd.to_datetime(df["start_hour"])
    return df

def upload_to_hopsworks(df: pd.DataFrame):
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME
    )

    fg = project.get_feature_store().get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["start_hour", "start_station_id"],
        description="Hourly ride counts for top 3 Citi Bike stations",
        event_time="start_hour",
        online_enabled=False
    )

    fg.insert(df, write_options={"wait_for_job": True})
    print("Data inserted into Hopsworks Feature Group.")

if __name__ == "__main__":
    df = load_hourly_timeseries_data()
    upload_to_hopsworks(df)
