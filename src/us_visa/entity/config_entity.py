import os
from us_visa.constants import *
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TIMESTAMP=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    
    pipeline_name=PIPELINE_NAME
    artifact_dir=os.path.join(ARTIFACT_DIR,
                              TIMESTAMP
                              )
    timestamp=TIMESTAMP

# Creating object of above class
training_pipeline_config=TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    
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
    
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, 
                                                DATA_TRANSFORMATION_DIR_NAME)
    
    transformed_train_filepath: str = os.path.join(data_transformation_dir, 
                                                    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                    DATA_TRAIN_FILENAME.replace("csv", "npy")) 
    
    transformed_test_filepath: str = os.path.join(data_transformation_dir, 
                                                   DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   DATA_TEST_FILENAME.replace("csv", "npy"))
    
    transformed_object_filepath: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                     DATA_TRANSFORMATION_OBJECT_FILENAME)




@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_filepath: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME)
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_filepath: str = MODEL_TRAINER_MODEL_CONFIG_FILEPATH
    
    
    
@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_prod_model_key_path: str = f"{MODEL_PUSHER_S3_KEY}/{S3_PRODUCTION_MODEL_NAME}"


@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILENAME  # I think we need to change this


@dataclass
class USvisaPredictorConfig:
    #pred_model_filepath: str = f"{MODEL_PUSHER_S3_KEY}/{S3_PRODUCTION_MODEL_NAME}"
    #model_bucket_name: str = MODEL_BUCKET_NAME
    #DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "data_transformer_object"
    #DATA_TRANSFORMATION_OBJECT_FILENAME="data_preprocessor.pkl"
    data_preprocessor_filepath_local : Path = Path(os.path.join(DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                                DATA_TRANSFORMATION_OBJECT_FILENAME))
    
    pred_model_filepath_local : Path = Path(os.path.join(LOCAL_PRODUCTION_MODEL_DIR,
                                                         S3_PRODUCTION_MODEL_NAME))
    





