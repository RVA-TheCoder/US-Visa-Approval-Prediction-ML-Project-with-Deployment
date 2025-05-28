import os , sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from pandas import DataFrame

from us_visa.utils.main_utils import read_yaml_file
from us_visa.entity.config_entity import USvisaPredictorConfig
from us_visa.entity.s3_estimator import USvisaEstimator   # need to use the ProductionModel/production_model.pkl

from us_visa.exception import USvisaException
from us_visa.logger import logging




class USvisaData:
    
    # Usvisa Data constructor : Input -> all features of the trained model for prediction
    def __init__(self,
                continent,
                education_of_employee,
                has_job_experience,
                requires_job_training,
                no_of_employees,
                region_of_employment,
                prevailing_wage,
                unit_of_wage,
                full_time_position,
                yr_of_estab
                ):
        
        try:
            self.yr_of_estab = yr_of_estab
            self.prevailing_wage = prevailing_wage
            self.no_of_employees = no_of_employees
            
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.region_of_employment = region_of_employment
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            
        except Exception as e:
            raise USvisaException(e, sys) from e


    def get_usvisa_data_as_dict(self):
        
        """
        This function returns a dictionary from USvisaData class input 
        """
        
        logging.info("Entered get_usvisa_data_as_dict method as USvisaData class inside src/us_visa/pipline/prediction_pipeline.py file")

        try:
            input_data_dict = {
                "yr_of_estab": [self.yr_of_estab],
                "prevailing_wage": [self.prevailing_wage],
                "no_of_employees": [self.no_of_employees],
                
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "region_of_employment": [self.region_of_employment],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                
            }

            logging.info("Created usvisa input data dict")

            logging.info("Exited get_usvisa_data_as_dict method as USvisaData class inside src/us_visa/pipline/prediction_pipeline.py file")

            return input_data_dict

        except Exception as e:
            raise USvisaException(e, sys) from e


    def get_usvisa_input_data_frame(self)-> DataFrame:
        
        """
        This function returns a DataFrame from USvisaData class input
        """
        try:
            
            usvisa_input_data_dict = self.get_usvisa_data_as_dict()
            return DataFrame(usvisa_input_data_dict)
        
        except Exception as e:
            raise USvisaException(e, sys) from e


class USvisaClassifier:
    
    def __init__(self,
                 prediction_pipeline_config: USvisaPredictorConfig = USvisaPredictorConfig(),) -> None:
        
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
            
        except Exception as e:
            raise USvisaException(e, sys)


    def get_data_preprocessor_n_pred_model(self):
        
        """
        This method returns the data preprocessor object and trained model object.
        """
        logging.info("Entered get_pred_model_n_data_preprocessor method of USvisaClassifier class inside src/us_visa/pipline/prediction_pipeline.py file")

        data_preprocessor_path_local = self.prediction_pipeline_config.data_preprocessor_filepath_local
        pred_model_path_local = self.prediction_pipeline_config.pred_model_filepath_local
        
        with open(data_preprocessor_path_local, "rb") as data_preprocessor_handle:
            data_preprocessor = pickle.load(data_preprocessor_handle)
        
        
        with open(pred_model_path_local, mode="rb") as pred_model_handle:
            prediction_model = pickle.load(pred_model_handle)
            
        logging.info("Exited get_pred_model_n_data_preprocessor method of USvisaClassifier class inside src/us_visa/pipline/prediction_pipeline.py file")
            
        return data_preprocessor, prediction_model
        
        
    def predict(self, dataframe) -> str:
        
        """
        This is the method of USvisaClassifier
        Returns: Prediction in string format
        """
        
        try:
            logging.info("Entered predict method of USvisaClassifier class inside src/us_visa/pipline/prediction_pipeline.py file")
            #model = USvisaEstimator(
            #    bucket_name=self.prediction_pipeline_config.model_bucket_name,
            #    model_path=self.prediction_pipeline_config.model_filepath,
            #)
            
            data_preprocessor, prediction_model = self.get_data_preprocessor_n_pred_model()
            
            X_prod = data_preprocessor.transform(dataframe)
            print("X_prod : ", X_prod)
            
            result = prediction_model.predict(X_prod)
            print("Production data prediction : ", result)
            #result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise USvisaException(e, sys)
        
        