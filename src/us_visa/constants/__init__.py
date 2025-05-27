import os
from datetime import date
from pathlib import Path


from urllib.parse import quote_plus
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Fetch and encode
username = os.getenv("MONGO_USERNAME")
password = quote_plus(os.getenv("MONGO_PASSWORD"))  # safely encode special characters

# Build MongoDB URI
#MONGODB_URL_KEY="MONGODB_CONNECTION_STRING"
MONGODB_URI_KEY = f"mongodb+srv://{username}:{password}@us-visa-ml-cluster.q9tznwq.mongodb.net/?retryWrites=true&w=majority&appName=US-VISA-ML-Cluster"


DATABASE_NAME="US_VISA"
COLLECTION_NAME="visa_data_collection"


PIPELINE_NAME="usvisa"
ARTIFACT_DIR="artifact"
DATA_FILENAME="usvisa.csv"
DATA_TRAIN_FILENAME="train.csv"
DATA_TEST_FILENAME="test.csv"

MODEL_FILENAME="model.pkl"  

TARGET_COLUMN="case_status"
CURRENT_YEAR=date.today().year
PREPROCESSING_OBJECT_FILENAME="data_preprocessor.pkl"
SCHEMA_FILEPATH=os.path.join("config","schema.yaml")


AWS_ACCESS_KEY_ID_ENV_KEY="AWS_ACCESS_KEY_ID"            # defined inside .env file
AWS_SECRET_ACCESS_KEY_ENV_KEY="AWS_SECRET_ACCESS_KEY"    # defined inside .env file
REGION_NAME="us-east-1"


# Data-Ingestion related constants
DATA_INGESTION_DIR_NAME="data_ingestion"
DATA_INGESTION_COLLECTION_NAME="visa_data_collection"
DATA_INGESTION_FEATURE_STORE_DIR="feature_store"
DATA_INGESTION_INGESTED_DIR="ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO=0.3


# Data-Validation related constants
DATA_VALIDATION_DIR_NAME:str="data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR:str="drift_report"
DATA_VALIDATION_COMPLETE_DRIFT_REPORT_FILENAME:str="complete_drift_report.yaml"
DATA_VALIDATION_BRIEF_DRIFT_REPORT_FILENAME:str="brief_drift_report.yaml"
DATA_VALIDATION_COLUMNWISE_DRIFT_REPORT_FILENAME:str="columnwise_drift_report.csv"
DATA_VALIDATION_HTML_DRIFT_REPORT_FILENAME:str="data_drift_stats.html"



# Data Transformation ralated constant 
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed_data"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "data_transformer_object"



# MODEL TRAINER related constant 
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "trained_model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILEPATH: str = os.path.join("config", "model.yaml")



# MODEL EVALUATION related constant 
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02   # i.e., 2% change
MODEL_BUCKET_NAME = "usvisa-mlproj-s3"      # AWS S3 bucket name
MODEL_PUSHER_S3_KEY = "production_model_registry"      # inside this folder our trained production model will be saved.
S3_PRODUCTION_MODEL_NAME="production_model.pkl"
LOCAL_PRODUCTION_MODEL_DIR=Path("ProductionModel")























