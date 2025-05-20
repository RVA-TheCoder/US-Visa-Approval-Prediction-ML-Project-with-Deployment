import sys

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact

from us_visa.components.data_ingestion import DataIngestion



class TrainPipeline:
    
    def __init__(self):
        
        self.data_ingestion_config=DataIngestionConfig()
        
        
    def start_data_ingestion(self)->DataIngestionArtifact:
        
        try:
            logging.info("Entered the start_data_ingestion method TrainPipeline class inside src/us_visa/pipline/training_pipeline.py")
            
            logging.info("Getting the data from mongodb")
            
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            
            logging.info("Got the train_data and test_data")
            logging.info("Exited the start_data_ingestion method TrainPipeline class inside src/us_visa/pipline/training_pipeline.py")
            
            return data_ingestion_artifact
        
        except Exception as e:
            raise USvisaException(e, sys) from e
        
    
    def run_pipeline(self) -> None:
        
        """
        Runs the complete pipeline.
        """
        
        try :
            data_ingestion_artifact=self.start_data_ingestion()
            
            
            
        except Exception as e:
            raise USvisaException(e, sys) from e
        
        
