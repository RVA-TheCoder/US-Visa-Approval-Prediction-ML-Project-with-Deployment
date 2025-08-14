import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.constants import NUM_ESTIMATORS, MAX_DEPTH, LEARNING_RATE

"""
This module contains :

1. TargetValueMapping class:
    - Encodes and decodes target labels ("Certified" -> 1, "Denied" -> 0).

2. USvisaModel class:
    - Encapsulates the creation of a Gradient Boosting Classifier with pre-defined
      or user-specified hyperparameters.
    - Used for training and inference in the US Visa Approval prediction pipeline.
"""




class TargetValueMapping:
    
    """
    Provides label encoding and decoding for the target variable.

    Encoding scheme:
        "Certified" → 1 (positive class)
        "Denied"    → 0 (negative class)
    """
    
    def __init__(self):
        self.Certified:int = 1     # Positive class
        self.Denied:int = 0        # Negative class
     
        
    def _asdict(self):
        
        """
        Converts the mapping object into a dictionary.

        Returns:
            dict: Keys are class names, values are numeric encodings.
        """
        
        return self.__dict__
    
    
    def reverse_mapping(self):
        
        """
        Creates a reverse mapping from encoded values to class labels.

        Returns:
            dict: Keys are numeric encodings, values are class names.
        """
        
        mapping_response = self._asdict()
        
        return dict(zip(mapping_response.values(),mapping_response.keys()))
    
    
    
class USvisaModel:
    
    """
        Encapsulates a Gradient Boosting Classifier for US Visa Approval prediction.

        Attributes:
            (a) model_name (str): Name of the model.
            (b) n_estimators (int): Number of boosting stages.
            (c) max_depth (int): Maximum depth of individual estimators.
            (d) learning_rate (float): Learning rate shrinks contribution of each tree.
            (e) gradboost_params (dict): Dictionary of hyperparameters for the model.
        """
    
    def __init__(self, 
                 n_estimators:int=NUM_ESTIMATORS, 
                 max_depth:int=MAX_DEPTH, 
                 learning_rate:float=LEARNING_RATE
                 ):
        
        """
        Initializes the USvisaModel with given or default hyperparameters.

        Parameters : 
            (a) n_estimators (int): Number of boosting stages to be run (default=300).
            (b) max_depth (int): Maximum depth of each decision tree (default=5).
            (c) learning_rate (float): Step size shrinkage (default=0.01).
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
        
        """
        Creates a Gradient Boosting Classifier object using stored parameters.

        Returns:
            GradientBoostingClassifier: A scikit-learn model instance ready for training.
        """
        
        gb_model = GradientBoostingClassifier(**self.gradboost_params)
        
        return gb_model
        
   
   
   
   
    