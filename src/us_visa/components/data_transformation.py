import sys

import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from us_visa.constants import TARGET_COLUMN, SCHEMA_FILEPATH, CURRENT_YEAR
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from us_visa.entity.estimator import TargetValueMapping



class DataTransformation:
    
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig
                 ):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
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
        Returns : Pandas DataFrame
        
        """
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise USvisaException(e, sys)

    
    def get_data_transformer_object(self) -> Pipeline:
        
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
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


            #logging.info("Initialize PowerTransformer")

            #transform_pipe = Pipeline(steps=[
            #                          ('transformer', PowerTransformer(method='yeo-johnson'))
            #                        ])
            
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
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            
            if self.data_validation_artifact.data_validation_status:
                
                logging.info("Starting data transformation")
                
                # calling method
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                # calling static method to read the data
                train_df = DataTransformation.read_data(filepath=self.data_ingestion_artifact.trained_filepath)
                test_df = DataTransformation.read_data(filepath=self.data_ingestion_artifact.test_filepath)


                # For Train DF : Creating X_df and y_df (pandas series)
                X_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                y_train = train_df[TARGET_COLUMN]
                
                drop_cols = self._schema_config['drop_columns']

                logging.info(f"drop the columns {drop_cols} (config/schema.yaml) of Training dataset")

                X_train_df = drop_columns(df=X_train_df, cols = drop_cols)
                
                # Target Label mapping : Certified : 1 & Denied : 0
                encoding_map = TargetValueMapping()._asdict()
                print("Target label integer encoding mapper : ",encoding_map)
                y_train = y_train.map(encoding_map)

                logging.info("Got train predictors X_train_df  and train target label y_train of Training dataset")
                

                # For Test DF : Creating X_df and y_df (pandas series)
                X_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                y_test = test_df[TARGET_COLUMN]


                #input_feature_test_df['company_age'] = CURRENT_YEAR-input_feature_test_df['yr_of_estab']
                #logging.info("Added company_age column to the Test dataset")

                X_test_df = drop_columns(df=X_test_df, cols = drop_cols)

                logging.info(f"drop the columns in {drop_cols} of Test dataset")

                # Target Label mapping : Certified : 1 & Denied : 0
                y_test = y_test.map(encoding_map)

                logging.info("Got test predictors X_test_df  and test target label y_test of Training dataset")

                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )

                X_train_arr = preprocessor.fit_transform(X_train_df)
                logging.info(
                    "Used the preprocessor object to fit transform the train features"
                )
                
                X_test_arr = preprocessor.transform(X_test_df)
                logging.info("Used the preprocessor object to transform the test features")

                #logging.info("Applying SMOTEENN on Training dataset")

                #smt = SMOTEENN(sampling_strategy="minority")

                #input_feature_train_final, target_feature_train_final = smt.fit_resample(
                #                                                        input_feature_train_arr, 
                #                                                        target_feature_train_df
                #                                                        )

                #logging.info("Applied SMOTEENN on training dataset")

                #logging.info("Applying SMOTEENN on testing dataset")

                # This is wrong. Donot apply sampling techniques to test DF
                #input_feature_test_final, target_feature_test_final = smt.fit_resample(
                #                            input_feature_test_arr, target_feature_test_df
                #                        )

                #logging.info("Applied SMOTEENN on testing dataset")

                train_arr = np.c_[
                    X_train_arr, np.array(y_train)
                ]

                test_arr = np.c_[
                    X_test_arr, np.array(y_test)
                ]
                logging.info("Created train array and test array")
                
                # Saving column Transformer Preprocessor object
                save_object(self.data_transformation_config.transformed_object_filepath, preprocessor)
                
                # Saaving the train and test preprocessed data as numpy arrays
                save_numpy_array_data(self.data_transformation_config.transformed_train_filepath, 
                                      array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_filepath, 
                                      array=test_arr)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_filepath=self.data_transformation_config.transformed_object_filepath,
                    transformed_train_filepath=self.data_transformation_config.transformed_train_filepath,
                    transformed_test_filepath=self.data_transformation_config.transformed_test_filepath
                )
                
                return data_transformation_artifact
            
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise USvisaException(e, sys) from e




















