import os
from us_visa.constants import *
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# Timestamp for uniquely naming artifact directories for each pipeline run
TIMESTAMP=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    
    """
    Configuration for the overall training pipeline.

    Attributes:
        (a) pipeline_name (str): Name of the pipeline.
        (b) artifact_dir (str): Directory to store all pipeline artifacts.
        (c) timestamp (str): Timestamp to ensure uniqueness of artifact directory.
    """
    
    pipeline_name=PIPELINE_NAME
    artifact_dir=os.path.join(ARTIFACT_DIR, TIMESTAMP )
                                                    
    timestamp=TIMESTAMP


# Creating an instance for use across the project
training_pipeline_config=TrainingPipelineConfig()



@dataclass
class DataIngestionConfig:
    
    """
    Configuration for data ingestion stage.

    Attributes:
        (a) data_ingestion_dir (str): Base directory for data ingestion artifacts.
        (b) feature_store_filepath (str): Path to store the raw feature data.
        (c) training_filepath (str): Path to store the processed training data.
        (d) testing_filepath (str): Path to store the processed testing data.
        (e) train_test_split_ratio (float): Ratio for splitting training and test data.
        (f) collection_name (str): Database collection name (if applicable).
    """
    
    data_ingestion_dir:str = os.path.join(training_pipeline_config.artifact_dir,
                                          DATA_INGESTION_DIR_NAME
                                         )
                                     
    
    feature_store_filepath:str = os.path.join(data_ingestion_dir,
                                              DATA_INGESTION_FEATURE_STORE_DIR,
                                              DATA_FILENAME
                                            )
    
    training_filepath:str = os.path.join(data_ingestion_dir,
                                         DATA_INGESTION_INGESTED_DIR,
                                         DATA_TRAIN_FILENAME,
                                       )
    
    testing_filepath:str = os.path.join(data_ingestion_dir,
                                       DATA_INGESTION_INGESTED_DIR,
                                       DATA_TEST_FILENAME
                                       )
    
    train_test_split_ratio:float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME
    
    
    

@dataclass
class DataValidationConfig:
    
    """
    Configuration for data validation stage.

    Attributes:
        (a) data_validation_dir (str): Base directory for validation artifacts.
        (b) html_drift_report_filepath (str): Path to HTML format drift report.
        (c) complete_drift_report_filepath (str): Path to complete drift report (JSON/other).
        (d) brief_drift_report_filepath (str): Path to brief summary drift report.
        (e) columnwise_drift_report_filepath (str): Path to per-column drift analysis.
    """
    
    data_validation_dir:str=os.path.join(training_pipeline_config.artifact_dir,
                                         DATA_VALIDATION_DIR_NAME
                                         )
    html_drift_report_filepath:str=os.path.join(data_validation_dir,
                                           DATA_VALIDATION_DRIFT_REPORT_DIR,
                                           DATA_VALIDATION_HTML_DRIFT_REPORT_FILENAME
                                           )
    
    complete_drift_report_filepath:str=os.path.join(data_validation_dir,
                                           DATA_VALIDATION_DRIFT_REPORT_DIR,
                                           DATA_VALIDATION_COMPLETE_DRIFT_REPORT_FILENAME
                                           )
    
    brief_drift_report_filepath:str=os.path.join(data_validation_dir,
                                           DATA_VALIDATION_DRIFT_REPORT_DIR,
                                           DATA_VALIDATION_BRIEF_DRIFT_REPORT_FILENAME
                                           )
    
    columnwise_drift_report_filepath:str=os.path.join(data_validation_dir,
                                           DATA_VALIDATION_DRIFT_REPORT_DIR,
                                           DATA_VALIDATION_COLUMNWISE_DRIFT_REPORT_FILENAME
                                           )


@dataclass
class DataTransformationConfig:
    
    """
    Configuration for data transformation stage.

    Attributes:
        (a) data_transformation_dir (str): Base directory for transformation artifacts.
        (b) transformed_train_data_filepath (str): Path for transformed training data (.npy).
        (c) transformed_test_data_filepath (str): Path for transformed test data (.npy).
        (d) data_transformer_object_filepath (str): Path to save transformation object (scaler, encoder, etc.).
    """
    
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, 
                                                DATA_TRANSFORMATION_DIR_NAME)
    
    transformed_train_data_filepath: str = os.path.join(data_transformation_dir, 
                                                    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                    DATA_TRAIN_FILENAME.replace("csv", "npy")) 
    
    transformed_test_data_filepath: str = os.path.join(data_transformation_dir, 
                                                   DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   DATA_TEST_FILENAME.replace("csv", "npy"))
    
    data_transformer_object_filepath: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                     DATA_TRANSFORMATION_OBJECT_FILENAME)
    
    

@dataclass
class ModelTrainerConfig:
    
    """
    Configuration for model training stage.

    Attributes:
        (a) model_trainer_dir (str): Directory for model training artifacts.
        (b) trained_model_filepath (str): Path to save trained model.
        (c) expected_accuracy (float): Minimum acceptable accuracy for model.
        (d) model_config_filepath (str): Path to model configuration file.
    """
    
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_filepath: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME)
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_filepath: str = MODEL_TRAINER_MODEL_CONFIG_FILEPATH
    
    
    
@dataclass
class ModelEvaluationConfig:
    
    """
    Configuration for model evaluation stage.

    Attributes:
        (a) changed_threshold_score (float): Minimum improvement needed to replace production model.
        (b) bucket_name (str): Cloud storage bucket name for model.
        (c) s3_prod_model_key_path (str): S3 key path for production model.
    """
    
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_prod_model_key_path: str = f"{MODEL_PUSHER_S3_KEY}/{S3_PRODUCTION_MODEL_NAME}"



@dataclass
class ModelPusherConfig:
    
    """
    Configuration for model pushing stage.

    Attributes:
        (a) bucket_name (str): Cloud storage bucket name.
        (b) s3_model_key_path (str): S3 key path to store pushed model.
    """
    
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILENAME  # TODO: Check correctness, might need adjustment



@dataclass
class USvisaPredictorConfig:
    
    """
    Configuration for local US Visa prediction service.

    Attributes:
        (a) data_preprocessor_filepath_local (Path): Local path to saved preprocessor object.
        (b) pred_model_filepath_local (Path): Local path to production model for predictions.
    """
    
    
    #pred_model_filepath: str = f"{MODEL_PUSHER_S3_KEY}/{S3_PRODUCTION_MODEL_NAME}"
    #model_bucket_name: str = MODEL_BUCKET_NAME
    #DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "data_transformer_object"
    #DATA_TRANSFORMATION_OBJECT_FILENAME="data_preprocessor.pkl"
    
    
    data_preprocessor_filepath_local : Path = Path(os.path.join(DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                                DATA_TRANSFORMATION_OBJECT_FILENAME))
    
    pred_model_filepath_local : Path = Path(os.path.join(LOCAL_PRODUCTION_MODEL_DIR,
                                                         S3_PRODUCTION_MODEL_NAME))
    





