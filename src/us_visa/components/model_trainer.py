import sys, os, json
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object

from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.artifact_entity import (DataTransformationArtifact,
                                            ModelTrainerArtifact, 
                                            ClassificationMetricArtifact)

from us_visa.entity.estimator import USvisaModel
from us_visa.constants import MODEL_PERFORMANCE_METRICS_FILENAME


class ModelTrainer:
    
    """
    Class responsible for training the machine learning model for US Visa Approval Prediction.

    This class handles:
      - Loading transformed training and testing datasets
      - Initializing the model
      - Training the model
      - Evaluating model performance
      - Saving the trained model
      - Creating and returning the ModelTrainerArtifact containing model file path and metrics
    """
    
    
    def __init__(self, 
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        
        """
        Initialize ModelTrainer with required artifacts and configurations.

        Parameters: 
            (a) data_transformation_artifact (DataTransformationArtifact):
                    Output from the Data Transformation stage containing transformed dataset file paths.
            (b) model_trainer_config (ModelTrainerConfig):
                    Configuration object containing model save paths and parameters.
        """
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config



    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        
        """
        Train the Gradient Boosting classifier and evaluate performance.

        Parameters : 
            (a) train (np.ndarray): Transformed training dataset (features + target as last column).
            (b) test (np.ndarray): Transformed testing dataset (features + target as last column).

        Returns:
            Tuple[object, ClassificationMetricArtifact]:
                - Trained Gradient Boosting model object.
                - Classification metrics for both train and test sets.
        """
        
        try:
            logging.info("Using neuro_mf to get best model object and report")
            
            US_visa_model=USvisaModel(n_estimators=300,
                                      max_depth=5,
                                      learning_rate=0.01)
            
            gb_model=US_visa_model.create_model_object()
            
            # Train Test splits
            X_train, y_train, X_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]


            # Fitting the model
            gb_model.fit(X_train, y_train)
            logging.info("Model Training completed")
            
            # Make predictions
            y_train_pred = gb_model.predict(X_train)
            y_test_pred = gb_model.predict(X_test)
           
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
            
            # Save metrics to JSON at project root
            metrics_dict = {
                "train": {
                    "accuracy": round(train_accuracy, 4),
                    "f1_score": round(train_f1_score, 4),
                    "precision": round(train_precision, 4),
                    "recall": round(train_recall, 4)
                },
                "test": {
                    "accuracy": round(test_accuracy, 4),
                    "f1_score": round(test_f1_score, 4),
                    "precision": round(test_precision, 4),
                    "recall": round(test_recall, 4)
                }
            }
            
            metrics_file_path = os.path.join(os.getcwd(), MODEL_PERFORMANCE_METRICS_FILENAME)
            with open(metrics_file_path, "w") as json_file:
                json.dump(metrics_dict, json_file, indent=4)
                
            logging.info(f"Model performance metrics saved to {metrics_file_path}")
            
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
        Execute the full model training pipeline:
          - Load transformed train & test data
          - Train the model
          - Evaluate model performance
          - Save trained model
          - Return ModelTrainerArtifact

        Returns:
            ModelTrainerArtifact: Artifact containing trained model path and performance metrics.
        """
        
        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class inside src/us_visa/components/model_trainer.py file")
            
            # Load transformed data arrays
            train_arr=load_numpy_array_data(filepath=self.data_transformation_artifact.transformed_train_data_filepath)
            test_arr=load_numpy_array_data(filepath=self.data_transformation_artifact.transformed_test_data_filepath)
            
            # Train model and get metrics
            trained_model, metric_artifacts = self.get_model_object_and_report(train=train_arr, 
                                                                               test=test_arr
                                                                               )
            logging.info("Model Training completed")
            
            # Save trained model to specified file path
            save_object(self.model_trainer_config.trained_model_filepath, trained_model)

            # Create artifact for model trainer stage
            model_trainer_artifact = ModelTrainerArtifact(
                                        trained_model_filepath=self.model_trainer_config.trained_model_filepath,
                                        metric_artifact=metric_artifacts,
                                     )
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact
        
        except Exception as e:
            raise USvisaException(e, sys) from e
        
        
        
        