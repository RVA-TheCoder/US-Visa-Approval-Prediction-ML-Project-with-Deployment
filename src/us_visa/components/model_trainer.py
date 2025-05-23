import sys
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from neuro_mf  import ModelFactory

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.artifact_entity import (DataTransformationArtifact,
                                            ModelTrainerArtifact, 
                                            ClassificationMetricArtifact)

from us_visa.entity.estimator import USvisaModel

class ModelTrainer:
    
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        
        """
        Method Name :   get_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        
        try:
            logging.info("Using neuro_mf to get best model object and report")
            
            US_visa_model=USvisaModel(n_estimators=300,
                                      max_depth=5,
                                      learning_rate=0.01)
            
            gb_model=US_visa_model.create_model_object()
            
            #model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            X_train, y_train, X_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            #best_model_detail = model_factory.get_best_model(
            #    X=x_train,y=y_train,base_accuracy=self.model_trainer_config.expected_accuracy
            #)
            #model_obj = best_model_detail.best_model

            # Fitting the model
            gb_model.fit(X_train, y_train)
            logging.info("Model Training completed")
            
            # Make predictions
            y_train_pred = gb_model.predict(X_train)
            y_test_pred = gb_model.predict(X_test)
            #y_pred = model_obj.predict(x_test)
            
            # Model Performance on train data
            train_accuracy = accuracy_score(y_train, y_train_pred) 
            train_f1_score = f1_score(y_train, y_train_pred)  
            train_precision = precision_score(y_train, y_train_pred)  
            train_recall = recall_score(y_train, y_train_pred)
            
            # Model performance on test data
            test_accuracy = accuracy_score(y_test, y_test_pred) 
            test_f1_score = f1_score(y_test, y_test_pred)  
            test_precision = precision_score(y_test, y_test_pred)  
            test_recall = recall_score(y_test, y_test_pred)
            
            print(f"Train_accuracy : {round(train_accuracy,4)*100}\nTest_accuracy : {round(test_accuracy,4)*100}" )
            print(f"Train_f1_score : {round(train_f1_score,4)*100}\nTest_f1_score : {round(test_f1_score,4)*100}" )
            print(f"Train_precision : {round(train_precision,4)*100}\nTest_precision : {round(test_precision,4)*100}" )
            print(f"Train_recall : {round(train_recall,4)*100}\nTest_recall : {round(test_recall,4)*100}\n" )
            
            logging.info("Model Prediction completed")
            
            metric_artifact = ClassificationMetricArtifact(train_accuracy=train_accuracy,
                                                           train_f1_score=train_f1_score,   # This is important for us because data is imbalanced
                                                           train_precision_score=train_precision,
                                                           train_recall_score=train_recall,
                                                           test_accuracy=test_accuracy,
                                                           test_f1_score=test_f1_score,     # This is important for us because data is imbalanced
                                                           test_precision_score=test_precision,
                                                           test_recall_score=test_recall
                                                           )
            
            return gb_model, metric_artifact
        
        except Exception as e:
            raise USvisaException(e, sys) from e
        

    def initiate_model_trainer(self, ) -> ModelTrainerArtifact:
        
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        
        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class inside src/us_visa/components/model_trainer.py file")
            
            train_arr=load_numpy_array_data(filepath=self.data_transformation_artifact.transformed_train_filepath)
            test_arr=load_numpy_array_data(filepath=self.data_transformation_artifact.transformed_test_filepath)
            
            trained_model, metric_artifacts = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model Training completed")
            
            #preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_filepath)


            #if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
            #    logging.info("No best model found with score more than base score")
            #    raise Exception("No best model found with score more than base score")

            #usvisa_model = USvisaModel(preprocessing_object=preprocessing_obj,
            #                           trained_model_object=best_model_detail.best_model)
            
            #logging.info("Created usvisa model object with preprocessor and model")
            #logging.info("Created best model file path.")
            
            save_object(self.model_trainer_config.trained_model_filepath, trained_model)

            model_trainer_artifact = ModelTrainerArtifact(
                                        trained_model_filepath=self.model_trainer_config.trained_model_filepath,
                                        metric_artifact=metric_artifacts,
                                    )
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact
        
        except Exception as e:
            raise USvisaException(e, sys) from e