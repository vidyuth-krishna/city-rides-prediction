import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

FEATURE_GROUP_NAME = "time_series_hourly_feature_group"
FEATURE_GROUP_VERSION = 1
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
TRANSFORMED_DATA_PATH = BASE_DIR / "data" / "transformed" / "top_3_hourly_timeseries.parquet"
FEATURE_VIEW_NAME = "citibike_hourly_features"
FEATURE_VIEW_VERSION = 1


MODELS_DIR = Path("models")
MODEL_NAME = "citibike_hourly_predictor"
MODEL_VERSION = 1

FEATURE_GROUP_MODEL_PREDICTION = "citibike_hourly_predictions"
