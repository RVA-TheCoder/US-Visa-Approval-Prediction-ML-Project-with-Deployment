import os
from datetime import date
from pathlib import Path



"""
This module centralizes all constant definitions for the US-Visa Approval ML project.

It stores values for:
    - Database details
    - File paths
    - AWS credentials keys
    - Model training thresholds
    - API configurations

Purpose:
Centralizing constants ensures that changes can be made in one place, improving
maintainability, consistency, and reducing hard-coded values across the codebase.
"""




# ============================
# MongoDB Database Details
# ============================
DATABASE_NAME="US_VISA"
COLLECTION_NAME="visa_data_collection"



# ============================
# General Pipeline Configurations
# ============================
PIPELINE_NAME="usvisa"
ARTIFACT_DIR="artifact"


# File names used across the pipeline
DATA_FILENAME="usvisa.csv"
DATA_TRAIN_FILENAME="train.csv"
DATA_TEST_FILENAME="test.csv"
MODEL_FILENAME="model.pkl"  

# Target column in dataset
TARGET_COLUMN="case_status"

# Current year (used for features like year of establishment)
CURRENT_YEAR=date.today().year

# Schema configuration file
SCHEMA_FILEPATH=os.path.join("config","schema.yaml")


# ============================
# AWS Cloud Configurations
# ============================
AWS_ACCESS_KEY_ID_ENV_KEY="AWS_ACCESS_KEY_ID"            # Environment variable in .env
AWS_SECRET_ACCESS_KEY_ENV_KEY="AWS_SECRET_ACCESS_KEY"    # Environment variable in .env
REGION_NAME="us-east-1"



# ============================
# Data Ingestion Configurations
# ============================
DATA_INGESTION_DIR_NAME="data_ingestion"
DATA_INGESTION_COLLECTION_NAME="visa_data_collection"
DATA_INGESTION_FEATURE_STORE_DIR="feature_store"
DATA_INGESTION_INGESTED_DIR="ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO=0.3  # 30% for test, 70% for train




# ============================
# Data Validation Configurations
# ============================
DATA_VALIDATION_DIR_NAME : str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR : str = "drift_report"
DATA_VALIDATION_COMPLETE_DRIFT_REPORT_FILENAME : str = "complete_drift_report.yaml"
DATA_VALIDATION_BRIEF_DRIFT_REPORT_FILENAME : str = "brief_drift_report.yaml"
DATA_VALIDATION_COLUMNWISE_DRIFT_REPORT_FILENAME : str = "columnwise_drift_report.csv"
DATA_VALIDATION_HTML_DRIFT_REPORT_FILENAME : str = "data_drift_stats.html"




# ============================
# Data Transformation Configurations
# ============================
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed_data"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "data_transformer_object"
DATA_TRANSFORMATION_OBJECT_FILENAME="data_preprocessor.pkl"



# ============================
# Model Trainer Configurations
# ============================
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "trained_model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILEPATH: str = os.path.join("config", "model.yaml")
MODEL_PERFORMANCE_METRICS_FILENAME : str = "model_performance_metrics.json"

# Gradient Boosted Model Parameters
NUM_ESTIMATORS : int = 300
MAX_DEPTH : int = 5
LEARNING_RATE : float = 0.01


# ============================
# Model Evaluation & Deployment Configurations
# ============================
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02    # 2% score change threshold
MODEL_BUCKET_NAME = "usvisa-mlproj-s3"                    # AWS S3 bucket for storing models
MODEL_PUSHER_S3_KEY = "production_model_registry"         # inside this folder our trained production model will be saved.
S3_PRODUCTION_MODEL_NAME="production_model.pkl"
LOCAL_PRODUCTION_MODEL_DIR=Path("ProductionModel")        # Local directory for production model



# ============================
# REST API Configurations
# ============================
APP_HOST = "0.0.0.0"
APP_PORT = 8080






