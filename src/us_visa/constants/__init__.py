import os
from datetime import date


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
PREPROCESSING_OBJECT_FILENAME="preprocessing.pkl"
SCHEMA_FILEPATH=os.path.join("config","schema.yaml")


AWS_ACCESS_KEY_ID_ENV_KEY="AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY="AWS_SECRET_ACCESS_KEY"
REGION_NAME="us-east-1"


# Data-Ingestion related constants
DATA_INGESTION_DIR_NAME="data_ingestion"
DATA_INGESTION_COLLECTION_NAME="visa_data_collection"
DATA_INGESTION_FEATURE_STORE_DIR="feature_store"
DATA_INGESTION_INGESTED_DIR="ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO=0.3


# Data-Validation related constants




