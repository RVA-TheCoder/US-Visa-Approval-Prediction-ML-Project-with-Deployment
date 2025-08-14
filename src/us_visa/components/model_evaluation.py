import sys
import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.constants import TARGET_COLUMN, CURRENT_YEAR
from us_visa.constants import * 

from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.artifact_entity import (DataIngestionArtifact,
                                            DataTransformationArtifact,
                                            ModelTrainerArtifact,
                                            ModelEvaluationArtifact)
                                            

from us_visa.entity.s3_estimator import USvisaEstimator

from us_visa.entity.estimator import USvisaModel
from us_visa.entity.estimator import TargetValueMapping



@dataclass
class EvaluateModelResponse:
    
    """
    Data structure to hold evaluation results comparing the trained model with 
    the best model from production (S3).
    """
    
    trained_model_f1_score: float
    best_model_f1_score: float
    is_trained_model_accepted: bool
    eval_metric_f1score_diff: float



class ModelEvaluation:
    
    """
    This class handles the evaluation of a newly trained model against the current 
    production model stored in S3.
    """

    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact,
                 model_eval_config: ModelEvaluationConfig, 
                 ):
        
        """
        Initialize ModelEvaluation with all required artifacts and config.
        
        Parameters : 
            (a) data_ingestion_artifact: Contains paths to raw train/test data.
            (b) data_transformation_artifact: Contains paths to transformed data arrays.
            (c) model_trainer_artifact: Contains paths and metrics of trained model.
            (d) model_eval_config: Contains evaluation configuration and S3 model paths.
        """
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_eval_config = model_eval_config
            
                 
        except Exception as e:
            raise USvisaException(e, sys) from e


    # from S3 bucket
    def get_best_model(self) -> Optional[USvisaEstimator]:
        
        """
        Retrieve the best (production) model from S3, if available.
        
        return: USvisaEstimator object if model exists, else None.
        """
        
        try:
            bucket_name = self.model_eval_config.bucket_name
            s3_model_path=self.model_eval_config.s3_prod_model_key_path
            
            usvisa_estimator_obj = USvisaEstimator(bucket_name=bucket_name,
                                               s3_prod_model_path=s3_model_path)

            #if usvisa_estimator.is_s3_model_present(model_path=model_path):
            if usvisa_estimator_obj.is_s3_model_present():
                return usvisa_estimator_obj.load_prod_model()
            
            return None
        
        except Exception as e:
            raise  USvisaException(e,sys)


    # compare the current trained model and S3 production model
    def evaluate_model(self) -> EvaluateModelResponse:
        
        """
        Compare the newly trained model with the current production model.
        
        Steps:
        - Load transformed test dataset.
        - Get trained model's F1-score.
        - If production model exists, evaluate it on test data.
        - Decide if the trained model should replace the production model.
        
        return: EvaluateModelResponse object containing comparison results.
        """
        
        try:
            
            test_data_array = np.load(self.data_transformation_artifact.transformed_test_data_filepath)
            
            X_test = test_data_array[:,:-1]
            y_test = test_data_array[:,-1]
            
            # trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.train_f1_score

            # Production model
            best_model_f1_score=None
            best_model = self.get_best_model()    # we have write our method
            
            if best_model is not None:
                
                # y_test_pred = s3_prod_model.predict(X_test)
                y_test_pred_best_model = best_model.predict(X_data=X_test)
                best_model_f1_score = f1_score(y_test, y_test_pred_best_model)
                
            else :
                print("Production Model is absent in S3 bucket")
                print("We need to push the current trained model to S3 bucket")
            
            
            if best_model_f1_score:
                
                if trained_model_f1_score > best_model_f1_score :
                    
                    changed_f1_score=round(trained_model_f1_score-best_model_f1_score,4) 
                    is_trained_model_accepted_for_S3_push=changed_f1_score> MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
                
                else :
                    is_trained_model_accepted_for_S3_push=False
                    
            else : 
                is_trained_model_accepted_for_S3_push=True
                changed_f1_score=0.0
            
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           
                                           is_trained_model_accepted=is_trained_model_accepted_for_S3_push,  
                                           eval_metric_f1score_diff=changed_f1_score
                                           )
            
            
            logging.info(f"Result: {result}")
            
            return result

        except Exception as e:
            raise USvisaException(e, sys)
        


    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        
        """
        Orchestrates the model evaluation process and returns an artifact.

        return: ModelEvaluationArtifact containing evaluation outcome and paths.
        """
        
        try:
            # calling the method
            evaluate_model_response = self.evaluate_model()
            s3_prod_model_path = self.model_eval_config.s3_prod_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_trained_model_accepted=evaluate_model_response.is_trained_model_accepted,
                s3_prod_model_path=s3_prod_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_filepath,
                eval_metric_f1score_diff=evaluate_model_response.eval_metric_f1score_diff
                )

            
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            
            return model_evaluation_artifact
        
        except Exception as e:
            raise USvisaException(e, sys) from e
        
        
        
        