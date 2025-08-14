import os, sys, json
import pandas as pd
from pandas import DataFrame

from evidently import Report
from evidently.presets import DataSummaryPreset , DataDriftPreset 

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa.constants import SCHEMA_FILEPATH

from us_visa.entity.config_entity import DataValidationConfig

from us_visa.entity.artifact_entity import (DataIngestionArtifact,
                                            DataValidationArtifact)



"""
This module contains the DataValidation class for the US Visa Approval prediction project.

It validates the ingested training and testing datasets against the defined schema and checks 
for data drift.

Validation steps include:
    - Verifying the number of columns
    - Checking required column existence
    - Detecting dataset drift using Evidently AI reports
"""


class DataValidation:
    
    """
    A class to validate datasets before further processing in the pipeline.
    """
    
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        
        """
        Initialize DataValidation object.

        Parameters : 
            (a) data_ingestion_artifact (DataIngestionArtifact): Paths for train and test data files.
            (b) data_validation_config (DataValidationConfig): Configurations for validation process.
        """
        
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml_file(filepath=SCHEMA_FILEPATH)
            
            
        except Exception as e:
            raise USvisaException(e, sys) from e
       
        
    @staticmethod
    def read_data(filepath)->DataFrame:
        
        """
        Reads a CSV file and returns it as a pandas DataFrame.

        Parameters : 
            (a) filepath (str): Path to CSV file.

        Returns:
            DataFrame: Loaded data.
        """
        
        try:
            return pd.read_csv(filepath)
        
        except Exception as e:
            raise USvisaException(e, sys) from e
        
      
      
    def validate_number_of_columns(self, dataframe:DataFrame) ->bool:
        
        """
        Validate that the number of columns in the dataframe matches the schema definition.

        Parameters : 
            dataframe (DataFrame): Input dataframe.

        Returns:
            bool: True if column count matches schema, else False.
        """
        
        try:
            status=len(dataframe.columns) == len(self._schema_config["columns"])
            
            logging.info(f"Is required number of columns present : [{status}]")
            
            return status
        
        except Exception as e:
            raise USvisaException(e, sys) from e
        
        
        
    def is_column_exist(self, df:DataFrame) ->bool :
        
        """
        Validate the existence of all numerical and categorical columns in the dataframe.

        Parameters : 
            (a) df (DataFrame): Input dataframe.

        Returns:
            bool: True if all required columns exist, else False.
        """
        
        try:
            dataframe_columns=df.columns
            missing_numerical_columns=[]
            missing_categorical_columns=[]
            
            # This self._schema_config["numerical_columns"] will return a list
            for column in self._schema_config["numerical_columns"]:
                   
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
            
            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical columns : {missing_numerical_columns}")
                print(f"Missing numerical columns : {missing_numerical_columns}\n")
             
                    
            for column in self._schema_config["categorical_columns"]:
                
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical columns : {missing_categorical_columns}")
                print(f"Missing categorical columns : {missing_categorical_columns}\n")

            status = len(missing_numerical_columns) == 0 and len(missing_categorical_columns) == 0
            return status
        
        except Exception as e:
            raise USvisaException(e, sys) from e
        
        

    def detect_dataset_drift(self,
                             reference_df:DataFrame,
                             current_df:DataFrame) ->bool :
        
        """
        Detect if there is a significant drift between reference (train) and current (test) datasets.

        Parameters : 
            (a) reference_df (DataFrame): Reference dataset (usually training data).
            (b) current_df (DataFrame): Current dataset (usually testing data).

        Returns:
            bool: True if drift detected, else False.
        """
        
        try:

            # PART 1 : Generate Evidently AI drift report 
            evidently_ai_report_obj = Report( [DataDriftPreset()], include_tests="True" )
            evidently_report_snapshort_obj = evidently_ai_report_obj.run(reference_data=reference_df, current_data=current_df)
            
            #print("evidently_report_snapshort_obj : " ,evidently_report_snapshort_obj)
            #print("type evidently_report_snapshort_obj : " , type(evidently_report_snapshort_obj))

            os.makedirs(os.path.dirname(self.data_validation_config.html_drift_report_filepath), exist_ok=True)
            evidently_report_snapshort_obj.save_html(self.data_validation_config.html_drift_report_filepath)

            # returns the json in str format
            report_str = evidently_report_snapshort_obj.json()        # need to change this method
            
            # Parse JSON drift report
            # Deserialize  (str, bytes or byte array instance containing a JSON document) to a Python object.
            json_report=json.loads(report_str)     
            write_yaml_file(filepath=self.data_validation_config.complete_drift_report_filepath,
                            content=json_report,
                            replace=True)


            # PART2 : Extract drift summary
            overall_drift_Status= {}
            columnwise_drift_report = {}
            
            for obj in json_report['tests'] :
            
                if obj['id'] == 'lt':

                    # Brief drift report
                    overall_drift_Status["name"] = obj['name']
                    overall_drift_Status["description"] = obj['description']
                    overall_drift_Status["data_validation_status"] = obj['status']
                    overall_drift_Status["threshold"] = obj['bound_test']['test']['threshold']
            
                else : 
                    
                    # Column wise report
                    columnwise_drift_report[obj['metric_config']['params']['column']] = [{"description" : obj['description'] , 
                                                                                           "status" : obj['status']
                                                                                         }]


            # Brief drift report
            if overall_drift_Status["data_validation_status"].lower() == "success" :
                overall_drift_Status['data_drift_status'] = False
                
            else :
                overall_drift_Status['data_drift_status'] = True
                
            print("Brief drift report : \n",overall_drift_Status)
            write_yaml_file(filepath=self.data_validation_config.brief_drift_report_filepath, 
                            content=overall_drift_Status, 
                            replace=True)


            # PART 3 : Save column-wise drift report as pandas DF
            # Convert dictionary to DataFrame
            df = pd.DataFrame.from_dict({k: v[0] for k, v in columnwise_drift_report.items()}, 
                                        orient='index')
            
            # Optional: Rename index name
            df.index.name = 'feature'  
            df.to_csv(self.data_validation_config.columnwise_drift_report_filepath)

            
            if overall_drift_Status["data_validation_status"].lower() == "success":
                drift_status=False

            else :
                drift_status=True
        
    
            return drift_status   # return type bool
        
        except Exception as e:
            raise USvisaException(e, sys) from e
         
        
        
    def initiate_data_validation(self)->DataValidationArtifact:
        
        """
        Initiates the data validation process.

        Steps:
            1. Validate column count for train and test datasets.
            2. Validate required column existence.
            3. Detect dataset drift.

        Returns:
            DataValidationArtifact: Results of validation process.
        """
        
        try:   
            
            data_validation_msg=""
            logging.info("Starting data Validation")
            
            train_df=DataValidation.read_data(filepath=self.data_ingestion_artifact.trained_data_filepath)
            test_df=DataValidation.read_data(filepath=self.data_ingestion_artifact.test_data_filepath)
            
            # Step 1: Column count validation : Checking for the total columns count both in Train and Test data
            validate_number_of_columns_status_train=self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"Is total columns count matches the required columns count in training dataframe : {validate_number_of_columns_status_train}")
            if not validate_number_of_columns_status_train:
                data_validation_msg+=f"Training dataframe column count mismatch. "
                
                
            validate_number_of_columns_status_test=self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"Is total columns count matches the required columns count in testing dataframe : {validate_number_of_columns_status_test}")
            if not validate_number_of_columns_status_test:
                data_validation_msg+=f"Testing dataframe column count mismatch. "
            
            
            # Step 2: Column existence validation : Checking for the required columns both in Train and Test data 
            is_column_exist_status_train = self.is_column_exist(df=train_df)
            logging.info(f"All required columns present in training dataframe : {is_column_exist_status_train}")
            if not is_column_exist_status_train:
                data_validation_msg+=f"Missing columns in training dataframe. "    
                
            
            is_column_exist_status_test = self.is_column_exist(df=test_df)
            logging.info(f"All required columns present in testing dataframe : {is_column_exist_status_test}")
            if not is_column_exist_status_test:
                data_validation_msg+=f"Missing columns in testing dataframe. "  

            # Will be used later with data drift status
            validate_number_of_columns_status=validate_number_of_columns_status_train and  validate_number_of_columns_status_test
            is_column_exist_status=is_column_exist_status_train and is_column_exist_status_test
            
        
            # We could change 'validation_status' to 'pre_validation_status'.
            pre_data_validation_status=len(data_validation_msg)==0
            
            # Step 3: Drift detection (only if pre-validation passed)
            if pre_data_validation_status:
                
                drift_status=self.detect_dataset_drift(reference_df=train_df, current_df=test_df)
                
                data_validation_msg += "Drift detected. " if drift_status else "No drift detected. "
                    
            else:
                logging.info(f"Data Validation message : {data_validation_msg}")
            
            # Complete data validation status as a boolean value   
            # data_validation_status = validate_number_of_columns_status and is_column_exist_status and (not drift_status)
            data_validation_status = pre_data_validation_status and not drift_status
            
            print("data_validation_status : ", data_validation_status)
            print("data_validation_message : ",data_validation_msg )
            
            data_validation_artifact=DataValidationArtifact(data_validation_status=data_validation_status,
                                                            data_validation_message=data_validation_msg,
                                                            data_drift_report_filepath=self.data_validation_config.complete_drift_report_filepath
                                                            )    
     
                
            logging.info(f"Data Validation artifact : {data_validation_artifact}")
            
            return data_validation_artifact
    
        except Exception as e:
            raise USvisaException(e, sys) from e
            
            
            
            
            
            
            
            
            
            
            
            