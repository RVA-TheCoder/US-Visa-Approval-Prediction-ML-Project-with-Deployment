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
    
    """
    Data container for US Visa prediction inputs.

    Accepts all the features required by the trained model for prediction
    and provides helper methods to convert them into dict or DataFrame formats.
    """
    
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
        
        """
        Initialize prediction input data.

        Parameters : 
            (a) continent: Continent of employment.
            (b) education_of_employee: Highest education level of the employee.
            (c) has_job_experience: 'Y' or 'N' indicating job experience.
            (d) requires_job_training: 'Y' or 'N' indicating if job training is required.
            (e) no_of_employees: Number of employees in the company.
            (f) region_of_employment: Region of employment within the continent.
            (g) prevailing_wage: Prevailing wage offered for the position.
            (h) unit_of_wage: Unit for wage (Year, Month, Week, Hour).
            (i) full_time_position: 'Y' or 'N' indicating if the position is full-time.
            (j) yr_of_estab: Year company was established.
        """
        
        try:
            # Store all provided feature values
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
        Returns:
            Dictionary containing model input features, each wrapped in a list
            (to match DataFrame row format for prediction).
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
        Returns:
            Pandas DataFrame containing one row of model input features.
        """
        try:
            
            usvisa_input_data_dict = self.get_usvisa_data_as_dict()
            return DataFrame(usvisa_input_data_dict)
        
        except Exception as e:
            raise USvisaException(e, sys) from e


class USvisaClassifier:
    
    """
    Class responsible for loading the preprocessing pipeline and trained model,
    and making predictions.
    """
    
    def __init__(self,
                 prediction_pipeline_config: USvisaPredictorConfig = USvisaPredictorConfig()
                 ) -> None:
        
        """
        Parameters : 
            (a) prediction_pipeline_config: Configuration object containing file paths
                                            for the preprocessor and prediction model.
        """
        
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
            
        except Exception as e:
            raise USvisaException(e, sys)


    def get_data_preprocessor_n_pred_model(self):
        
        """
        Loads:
            - Data preprocessor object (for transforming raw features).
            - Trained prediction model.

        Returns:
            Tuple (data_preprocessor, prediction_model)
        """
        
        logging.info("Entered get_pred_model_n_data_preprocessor method of USvisaClassifier class inside src/us_visa/pipline/prediction_pipeline.py file")

        data_preprocessor_path_local = self.prediction_pipeline_config.data_preprocessor_filepath_local
        pred_model_path_local = self.prediction_pipeline_config.pred_model_filepath_local
        
        # Load preprocessing pipeline
        with open(data_preprocessor_path_local, "rb") as data_preprocessor_handle:
            data_preprocessor = pickle.load(data_preprocessor_handle)
        
        # Load trained model
        with open(pred_model_path_local, mode="rb") as pred_model_handle:
            prediction_model = pickle.load(pred_model_handle)
            
        logging.info("Successfully loaded preprocessor and prediction model.\nExited get_pred_model_n_data_preprocessor method of USvisaClassifier class inside src/us_visa/pipline/prediction_pipeline.py file")
            
        return data_preprocessor, prediction_model
        
        
    def predict(self, dataframe) -> str:
        
        """
        Runs prediction on the provided DataFrame.

        Parameters : 
            (a) dataframe: Input data for prediction (must match model features after preprocessing).

        Returns:
            Numpy array containing predictions.
        """
        
        try:
            logging.info("Entered predict method of USvisaClassifier class inside src/us_visa/pipline/prediction_pipeline.py file")
            
            # Load the preprocessor and model
            data_preprocessor, prediction_model = self.get_data_preprocessor_n_pred_model()
            
            # Transform input data
            X_prod = data_preprocessor.transform(dataframe)
            print("X_prod : ", X_prod)
            
            # Make prediction
            result = prediction_model.predict(X_prod)
            print("Production data prediction : ", result)
            
            return result
        
        except Exception as e:
            raise USvisaException(e, sys)
        
        