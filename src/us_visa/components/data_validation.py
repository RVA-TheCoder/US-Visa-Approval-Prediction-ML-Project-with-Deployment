import os, sys, json
import pandas as pd
from pandas import DataFrame

from evidently import Report
from evidently.presets import DataSummaryPreset , DataDriftPreset 

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file

from us_visa.entity.config_entity import DataValidationConfig

from us_visa.entity.artifact_entity import (DataIngestionArtifact,
                                            DataValidationArtifact)


from us_visa.constants import SCHEMA_FILEPATH



class DataValidation:
    
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml_file(filepath=SCHEMA_FILEPATH)
            
            
        except Exception as e:
            raise USvisaException(e, sys) from e
        
    @staticmethod
    def read_data(filepath)->DataFrame:
        
        try:
            return pd.read_csv(filepath)
        
        except Exception as e:
            raise USvisaException(e, sys) from e
        
      
    def validate_number_of_columns(self, dataframe:DataFrame) ->bool:
        
        """
        This methos validate the number of columns.
        
        Returns : returns the bool value based on validation results.
        """
        try:
            status=len(dataframe.columns) == len(self._schema_config["columns"])
            
            logging.info(f"Is required number of columns present : [{status}]")
            
            return status
        
        except Exception as e:
            raise USvisaException(e, sys) from e
        
        
        
    def is_column_exist(self, df:DataFrame) ->bool :
        
        """
        This method validates the existence of numerical and categorical columns.
        
        Returns : returns bool value based on validation results.
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
        This Method validates if drift is detected.
        
        Returns : returns bool value based on validation results.
        """
        try:
            
            report = Report( [DataDriftPreset()], include_tests="True" )
            
            my_eval = report.run(reference_data=reference_df, current_data=current_df)
            
            os.makedirs(os.path.dirname(self.data_validation_config.html_drift_report_filepath), exist_ok=True)
            my_eval.save_html(self.data_validation_config.html_drift_report_filepath)
            
            # returns a string object 
            my_eval_json= my_eval.json()        # can we change this method
            
            json_report=json.loads(my_eval_json)
            
            write_yaml_file(filepath=self.data_validation_config.complete_drift_report_filepath,
                            content=json_report,replace=True)
            
            # Brief drift report
            overall_results_dict={"total columns count":12 , 
                                  "total drift columns count":None, 
                                  "threshold drift share":0.50,
                                  "observed drift share":None,
                                  "drift status":None
                                    }
            
            overall_results_dict["total drift columns count"] = int(json_report['metrics'][0]['value']['count'])
            overall_results_dict["observed drift share"] = round(json_report['metrics'][0]['value']['share'],5)

            if overall_results_dict["observed drift share"] > overall_results_dict["threshold drift share"]:
                overall_results_dict['drift status'] =True
                
            else :
                overall_results_dict['drift status'] = False
                
            print("Brief drift report : \n",overall_results_dict)
            write_yaml_file(filepath=self.data_validation_config.brief_drift_report_filepath, 
                            content=overall_results_dict, replace=True)
                            
            # total number of columns in the DF including target label
            n_features=len(self._schema_config["columns"])
            # total number of drifted columns in the DF
            n_drifted_features=overall_results_dict['total drift columns count']
            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            
            
            # Column wise report
            df_dict = {"feature":[], "drift status":[] }
            for key in json_report['tests'][1:]:

                col_name= key['metric_config']['params']['column']
                col_status = key['status']

                if col_status.lower()=="success":
                    col_drift_status=False

                else:
                    col_drift_status=False

                df_dict["feature"].append(col_name)
                df_dict['drift status'].append(col_drift_status)
                
            columnwise_drift_report_df=pd.DataFrame(df_dict)
            columnwise_drift_report_df.to_csv(self.data_validation_config.columnwise_drift_report_filepath)
            
            overall_status=json_report['tests'][0]['status']
            print("Overall DF drift status : ",overall_status)

            if overall_status.lower() == "success":

                drift_status=False

            else :
                drift_status=True
            
            #drift_status=json_report['data_drift']['data']['metrics']['dataset_drift']
    
            return drift_status   # return type bool
        
        except Exception as e:
            raise USvisaException(e, sys) from e
        
        
    def initiate_data_validation(self)->DataValidationArtifact:
        
        """
        This method initiate the data validation component for the pipeline
        
        Returns : bool value based on the validation results.
        
        """
        try:   
            # we could change the 'validation_error_msg' to 'data_validation_msg'
            data_validation_msg=""
            logging.info("Starting data Validation")
            
            train_df=DataValidation.read_data(filepath=self.data_ingestion_artifact.trained_filepath)
            test_df=DataValidation.read_data(filepath=self.data_ingestion_artifact.test_filepath)
            
            # First : Checking for the total columns count both in Train and Test data
            validate_number_of_columns_status_train=self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"Is total columns count matches the required columns count in training dataframe : {validate_number_of_columns_status_train}")
            if not validate_number_of_columns_status_train:
                data_validation_msg+=f"total columns count doenot match the required columns count in training dataframe."
                
                
            validate_number_of_columns_status_test=self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"Is total columns count matches the required columns count in testing dataframe : {validate_number_of_columns_status_test}")
            if not validate_number_of_columns_status_test:
                data_validation_msg+=f"total columns count doenot match the required columns count in testing dataframe."
            
            
            # Second : Checking for the required columns both in Train and Test data 
            is_column_exist_status_train = self.is_column_exist(df=train_df)
            logging.info(f"All required columns present in training dataframe : {is_column_exist_status_train}")
            if not is_column_exist_status_train:
                data_validation_msg+=f"Columns are missing in training dataframe."    
                
            
            is_column_exist_status_test = self.is_column_exist(df=test_df)
            logging.info(f"All required columns present in testing dataframe : {is_column_exist_status_test}")
            if not is_column_exist_status_test:
                data_validation_msg+=f"Columns are missing in testing dataframe."  

            # Will be used later with data drift status
            validate_number_of_columns_status=validate_number_of_columns_status_train and  validate_number_of_columns_status_test
            is_column_exist_status=is_column_exist_status_train and is_column_exist_status_test
            
        
            # We could change 'validation_status' to 'pre_validation_status'.
            pre_data_validation_status=len(data_validation_msg)==0
            
            if pre_data_validation_status:
                
                drift_status=self.detect_dataset_drift(reference_df=train_df, current_df=test_df)
                
                if drift_status:
                    logging.info("Drift detected.")
                    # This could be validation_error_msg += validation_error_msg for complete summary
                    data_validation_msg+="Drift detected"
                    
                else:
                    # This could be validation_error_msg += validation_error_msg for complete summary
                    data_validation_msg+="Drift not detected"
                    
            else:
                logging.info(f"Data Validation message : {data_validation_msg}")
            
            # Complete data validation status as a boolean value   
            data_validation_status = validate_number_of_columns_status and is_column_exist_status and (not drift_status)
            
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
            
            
            
            
            
            
            
            
            
            
            
            