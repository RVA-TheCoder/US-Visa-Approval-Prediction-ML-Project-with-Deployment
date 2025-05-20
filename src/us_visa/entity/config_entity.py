import os
from us_visa.constants import *
from dataclasses import dataclass
from datetime import datetime

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
    
    











