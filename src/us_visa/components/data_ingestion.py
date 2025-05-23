import os, sys
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from us_visa.constants import *
from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.data_access.usvisa_data import USvisaData


class DataIngestion:
    
    def __init__(self, data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        
        try:
            self.data_ingestion_config=data_ingestion_config
            
        except Exception as e:
            
            raise USvisaException(e, sys) from e
        
    
    def export_data_into_feature_store(self)->DataFrame:
        
        """
        This method exports data from mongodb server to csv file in our local system
        """
        try:
            
            logging.info(f"Exporting data from mongodb")
            
            usvisa_data=USvisaData()
            dataframe=usvisa_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            
            logging.info(f"Shape of the dataframe : {dataframe.shape}")
            
            feature_store_filepath=self.data_ingestion_config.feature_store_filepath
            dir_path=os.path.dirname(feature_store_filepath)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Saving exported data into feature store filepath : {feature_store_filepath}")
            
            dataframe.to_csv(feature_store_filepath, index=False, header=True)
            
            return dataframe
        
        except Exception as e:
            raise USvisaException(e, sys) from e
        
        
    def split_data_as_train_test(self, dataframe:DataFrame)->None:
        
        """
        This method split the data into train and test data based on the train-test split ratio provided.
        
        """ 
        logging.info("Entered split_data_as_train_test method of DataIngestion class inside src/us_visa/components/data_ingestion.py")
        
        try:
            
            
            train_data, test_data = train_test_split(dataframe,
                                                     test_size=self.data_ingestion_config.train_test_split_ratio,
                                                     stratify=dataframe[TARGET_COLUMN])
            
            #train_data, test_data = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            print(f"Original Data shape : {dataframe.shape}\nTrain data shape : {train_data.shape}\nTest data shape : {test_data.shape}\n")
            
            logging.info("Performed train test split on the dataframe.")
            
            dir_path=os.path.dirname(self.data_ingestion_config.training_filepath)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Exporting train and test filepaths.")
            train_data.to_csv(self.data_ingestion_config.training_filepath, index=False, header=True)
            test_data.to_csv(self.data_ingestion_config.testing_filepath, index=False, header=True)
            
            logging.info(f"Exported train and test filepath.")

            logging.info("Exited split_data_as_train_test method of DataIngestion class inside src/us_visa/components/data_ingestion.py")
            
        except Exception as e:
            raise USvisaException(e, sys) from e
        
        
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        
        """
        This method initiates the data ingestion components of training pipeline.
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class inside src/us_visa/components/data_ingestion.py")
        
        try:
            dataframe=self.export_data_into_feature_store()
            logging.info("Got the data from MongoDB server.")
            
            self.split_data_as_train_test(dataframe)

            logging.info("Performed train-test split on the dataset.")
            
            logging.info("Exited initiate_data_ingestion method of DataIngestion class inside src/us_visa/components/data_ingestion.py")

            data_ingestion_artifact=DataIngestionArtifact(trained_filepath=self.data_ingestion_config.training_filepath,
                                                          test_filepath=self.data_ingestion_config.testing_filepath
                                                          )
            
            
            logging.info(f"Data Ingestion Artifact : {data_ingestion_artifact}")
            
            return data_ingestion_artifact
        
        except Exception as e:
            
            raise USvisaException(e, sys) from e
        
    
      
    
    
    
    
    
    
    
    
    
    
    
    