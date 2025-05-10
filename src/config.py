import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

FEATURE_GROUP_NAME = "time_series_hourly_feature_group"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "citibike_hourly_features"
FEATURE_VIEW_VERSION = 1


# MODEL_NAME = "taxi_demand_predictor_next_hour"
# MODEL_VERSION = 1

# FEATURE_GROUP_MODEL_PREDICTION = "taxi_hourly_model_prediction"