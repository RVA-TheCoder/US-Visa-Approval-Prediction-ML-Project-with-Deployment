import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from us_visa.exception import USvisaException
from us_visa.logger import logging



class TargetValueMapping:
    
    def __init__(self):
        self.Certified:int = 1  # Positive class
        self.Denied:int = 0   # Negative class
        
    def _asdict(self):
        return self.__dict__
    
    def reverse_mapping(self):
        
        mapping_response = self._asdict()
        
        return dict(zip(mapping_response.values(),mapping_response.keys()))
    
    
class USvisaModel:
    
    def __init__(self, n_estimators:int=300, max_depth:int=5, learning_rate:float=0.01):
        
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        # Based on Notebook experiment
        self.model_name="GradientBoosted_Classifier"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.gradboost_params = {'n_estimators': self.n_estimators,
                                 'max_depth': self.max_depth, 
                                 'learning_rate': self.learning_rate }

    def create_model_object(self):
        
        gb_model = GradientBoostingClassifier(**self.gradboost_params)
        
        return gb_model
        
    # def predict(self, dataframe: DataFrame) -> DataFrame:
        
    #     """
    #     Function accepts raw inputs and then transformed raw input using preprocessing_object
    #     which guarantees that the inputs are in the same format as the training data
    #     At last it performs prediction on transformed features
    #     """
    #     logging.info("Entered predict method of UTruckModel class")

    #     try:
    #         logging.info("Using the trained model to get predictions")

    #         transformed_feature = self.preprocessing_object.transform(dataframe)

    #         logging.info("Used the trained model to get predictions")
    #         return self.trained_model_object.predict(transformed_feature)

    #     except Exception as e:
    #         raise USvisaException(e, sys) from e

    # def __repr__(self):
    #     return f"{type(self.trained_model_object).__name__}()"

    # def __str__(self):
    #     return f"{type(self.trained_model_object).__name__}()"