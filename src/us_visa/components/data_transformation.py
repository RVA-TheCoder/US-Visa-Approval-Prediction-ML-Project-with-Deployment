import sys,os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from us_visa.constants import *
from us_visa.utils.main_utils import (save_object, 
                                      save_numpy_array_data,
                                      read_yaml_file, 
                                      drop_columns
                                      )

from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import (DataTransformationArtifact, 
                                            DataIngestionArtifact, 
                                            DataValidationArtifact
                                            )

from us_visa.exception import USvisaException
from us_visa.logger import logging


from us_visa.entity.estimator import TargetValueMapping



class DataTransformation:
    
    """
    Handles data preprocessing for the US Visa Approval Prediction project.

    This class:
      1. Reads ingested train/test CSV data.
      2. Cleans datasets (drops unused columns, encodes target labels).
      3. Builds a preprocessing pipeline (numeric scaling, one-hot encoding).
      4. Saves transformed datasets and preprocessor object for future use.

    Attributes:
        (a) data_ingestion_artifact (DataIngestionArtifact): Artifact from data ingestion step.
        (b) data_validation_artifact (DataValidationArtifact): Artifact from data validation step.
        (c) data_transformation_config (DataTransformationConfig): Config with transformation paths/settings.
        (d) _schema_config (dict): Column metadata read from schema.yaml file.
    """
    
    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig
                 ):
        
        """
        Initialize the DataTransformation object.

        Parameters : 
            (a) data_ingestion_artifact (DataIngestionArtifact): Output of data ingestion stage.
            (b) data_validation_artifact (DataValidationArtifact): Output of data validation stage.
            (c) data_transformation_config (DataTransformationConfig): Configuration for transformation.
        """
        try:
            
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            
            self._schema_config = read_yaml_file(filepath=SCHEMA_FILEPATH)
            
        except Exception as e:
            raise USvisaException(e, sys) from e


    @staticmethod
    def read_data(filepath) -> pd.DataFrame:
        
        """
        Reads a CSV file into a pandas DataFrame.

        Parameters : 
            (a) filepath (str or Path): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded dataframe.

        Raises:
            USvisaException: If CSV reading fails.
        """
        
        try:
            return pd.read_csv(filepath)
        
        except Exception as e:
            raise USvisaException(e, sys)

    
    def get_data_transformer_object(self) -> Pipeline:
        
        """
        Creates a preprocessing pipeline for training and inference.

        Returns:
            ColumnTransformer: A scikit-learn ColumnTransformer object.

        Notes:
            - Numeric features: scaled with StandardScaler.
            - Categorical features: encoded with OneHotEncoder (drop first category to avoid dummy trap).
        """
        
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class inside src/us_visa/components/data_transformation.py"
        )

        try:
            logging.info("Got numerical cols from schema config")

            ohe_features = self._schema_config['ohe_features']
            #ord_columns = self._schema_config['ord_columns']
            #transform_columns = self._schema_config['transform_columns']
            
            num_features = self._schema_config['num_features']

            # Create Column Transformer with 2 types of transformers
            numeric_transformer = StandardScaler()
            ohe_transformer = OneHotEncoder(drop='first')

            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder Transformers")

            
            preprocessor = ColumnTransformer(
                                [
                                    ("OneHotEncoder", ohe_transformer, ohe_features),
                                    ("StandardScaler", numeric_transformer, num_features) ,
                                ]
                            )
                                    
            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            
            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys) from e
        

    def initiate_data_transformation(self, ) -> DataTransformationArtifact:
        
        """
        Executes the data transformation workflow:
            1. Reads train/test data from ingestion stage.
            2. Cleans and prepares features/target variables.
            3. Applies preprocessing transformations.
            4. Saves transformed data and preprocessing object.

        Returns:
            DataTransformationArtifact: Paths to transformed datasets and preprocessor object.
        """
        
        try:
            
            # Before doing data transformation, checking 'data validation status'
            if self.data_validation_artifact.data_validation_status:
                
                logging.info("Starting data transformation")
                
                ############################ TRAIN DATA CLEANING #################################################

                # calling static method to read the training data
                train_df = DataTransformation.read_data(filepath=self.data_ingestion_artifact.trained_data_filepath)
                
                # For Train DF : Creating X_df and y_df (pandas series)
                X_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                
                # drop_columns : method form utils module
                drop_cols = self._schema_config['drop_columns']
                X_train_df = drop_columns(df=X_train_df, cols = drop_cols)
                
                logging.info(f"drop the columns {drop_cols} (config/schema.yaml) of Training dataset")

                
                y_train = train_df[TARGET_COLUMN]
                # Target Label mapping : Certified : 1 & Denied : 0
                encoding_map = TargetValueMapping()._asdict()
                print("Target label integer encoding mapper : ",encoding_map)
                y_train = y_train.map(encoding_map)

                logging.info("Got train predictors 'X_train_df'  and train target label 'y_train' of Training dataset")
                
                
                #################################### TEST DATA CLEANING #####################################
                
                # calling static method to read the test data
                test_df = DataTransformation.read_data(filepath=self.data_ingestion_artifact.test_data_filepath)
                
                # For Test DF : Creating X_df and y_df (pandas series)
                X_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                X_test_df = drop_columns(df=X_test_df, cols = drop_cols)
                
                logging.info(f"drop the columns in {drop_cols} of Test dataset")
                
                #input_feature_test_df['company_age'] = CURRENT_YEAR-input_feature_test_df['yr_of_estab']
                #logging.info("Added company_age column to the Test dataset")

                y_test = test_df[TARGET_COLUMN]
                # Target Label mapping : Certified : 1 & Denied : 0
                y_test = y_test.map(encoding_map)

                logging.info("Got test predictors X_test_df  and test target label y_test of Training dataset")

                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )

                #################### DATA TRANSFORMATION : COLUMN TRANSFORMER PREPROCESSOR ####################
                
                # calling method
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                X_train_arr = preprocessor.fit_transform(X_train_df)
                logging.info(
                    "Used the preprocessor object to fit transform the train features"
                )
                
                X_test_arr = preprocessor.transform(X_test_df)
                logging.info("Used the preprocessor object to transform the test features")

                
                train_arr = np.c_[
                    X_train_arr, np.array(y_train)
                ]

                test_arr = np.c_[
                    X_test_arr, np.array(y_test)
                ]
                logging.info("Created train array and test array")
                
                
                ################### SAVING TRANSFORMED DATA & DATA PREPROCESSOR OBJECT #################################
                
                # Saving column Transformer Preprocessor object (inside 'artifact/06_14_2025_11_36_22/data_transformation/data_transformer_object/data_preprocessor.pkl')
                save_object(self.data_transformation_config.data_transformer_object_filepath, preprocessor)
                
                # Saving the data Preprocessor object for prediction pipeline ('data_transformer_object/data_preprocessor.pkl')
                data_preprocessor_object_filepath = os.path.join(DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                                 DATA_TRANSFORMATION_OBJECT_FILENAME)
                
                os.makedirs(Path(os.path.dirname(data_preprocessor_object_filepath)), exist_ok=True)
                with open(Path(data_preprocessor_object_filepath), "wb") as data_preprocessor_handle:
                    pickle.dump(obj=preprocessor, file=data_preprocessor_handle)
                    
                print(f"Data Preprocessor successfully saved to {data_preprocessor_object_filepath}")
                logging.info("Saved the data preprocessor object")
                
                # Saving the train and test preprocessed data as numpy arrays
                save_numpy_array_data(self.data_transformation_config.transformed_train_data_filepath, 
                                      array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_data_filepath, 
                                      array=test_arr)

                logging.info("Saved the transformed data.")


                ####################### DATA TRANSFORMATION ARTIFACTS ####################################

                data_transformation_artifact = DataTransformationArtifact(
                    data_transformer_object_filepath=self.data_transformation_config.data_transformer_object_filepath,
                    transformed_train_data_filepath=self.data_transformation_config.transformed_train_data_filepath,
                    transformed_test_data_filepath=self.data_transformation_config.transformed_test_data_filepath
                )
                
                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )    
                
                return data_transformation_artifact
            
            else:
                raise Exception(self.data_validation_artifact.data_validation_message)

        except Exception as e:
            raise USvisaException(e, sys) from e






