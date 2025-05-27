import sys
from pathlib import Path
from typing import Optional, Union
from pandas import DataFrame
import numpy as np

from us_visa.constants import *
from us_visa.cloud_storage.aws_storage import SimpleStorageService
from us_visa.entity.estimator import USvisaModel
from us_visa.exception import USvisaException


class USvisaEstimator :
    
    """
    This class is used to save and retrieve us_visa model in and from s3 bucket and to do the prediction
    """

    def __init__(self, bucket_name, s3_prod_model_path=f"{MODEL_PUSHER_S3_KEY}/{S3_PRODUCTION_MODEL_NAME}"):
        
        """
        Parameters : 
         (a) bucket_name: Name of the bucket where model resides
         (b) model_path : Location of our model in bucket
        
        """
        
        self.bucket_name = bucket_name
        self.s3_model_path = s3_prod_model_path   # need to work on this f"{MODEL_PUSHER_S3_KEY}/{S3_PRODUCTION_MODEL_NAME}"
        self.s3 = SimpleStorageService()
        
        #self.loaded_model:USvisaModel=None
        self.loaded_model=None


    def is_s3_model_present(self,):
        
        """
        we can use below methods :
        
        (i) s3_file_exists(bucket_name=MODEL_BUCKET_NAME, 
                           s3_key=f"{MODEL_PUSHER_S3_KEY}/production_model.pkl") : bool
        (ii) get_s3_file_objects(filename=f"{MODEL_PUSHER_S3_KEY}/my_production_model.pkl", 
                                  bucket_name=MODEL_BUCKET_NAME) 
        (iii) self.loaded_model= download_s3_fileobject_as_file(object_summaries=file_objects1, destination_dir=Path("ProductionModel"))
        
        
        """
        
        try:
            
            # need to replace with our own method 
            #return self.s3.s3_key_path_available(bucket_name=self.bucket_name, 
            #                                     s3_key=self.s3_model_path)
            
            return self.s3.is_s3_file_exists(bucket_name=self.bucket_name, 
                                             s3_key=self.s3_model_path)
        
        except USvisaException as e: 
            print(e)
            
            return False


    # Need to change this
    # def load_model(self,)->USvisaModel:
    def load_prod_model(self, ):
        
        """
        we can use below methods :
        (i) is_model_present(self, model_path) : bool 
        (ii) get_s3_file_objects(filename=f"{MODEL_PUSHER_S3_KEY}/my_production_model.pkl", 
                                  bucket_name=MODEL_BUCKET_NAME) 
        (iii) self.loaded_model= download_s3_fileobject_as_file(object_summaries=file_objects1, destination_dir=Path("ProductionModel"))
        
        """
        
        
        # replaced self.s3.load_model with self.s3.load_n_save_prod_model
        #return self.s3.load_n_save_prod_model(self.model_path, bucket_name=self.bucket_name)
        
        return self.s3.load_n_save_prod_model( 
                                        s3_filename=self.s3_model_path,
                                        bucket_name=self.bucket_name, 
                                        local_destination_dir=Path(LOCAL_PRODUCTION_MODEL_DIR)
                                      )
        

    # We already defined the methods get_s3_file_objects and download_s3_fileobject_as_file
    # we use above methods to save the model in our local system
    # renamed 'save_model'  to 'upload_model_to_s3' method
    def upload_model_to_s3(self, from_filepath, remove:bool=False)->None:
        
        """
        Save the model to the model_path
        Parameters :
          (a) from_filepath: our local system model path
          (b) remove: By default it is False that means we'll have our model locally available in our system folder
        
        """
        try:
            
            
            self.s3.upload_file(from_filename=from_filepath,
                                bucket_name=self.bucket_name,
                                s3_key=self.s3_model_path,
                                remove=remove
                                )
            
            
            
            
        except Exception as e:
            raise USvisaException(e, sys)


    # we can create ProductionModel/production_model.pkl object inside our project
    # Then we no need to grab the production model present in the s3 bucket.
    def predict(self, X_data : Optional[Union[DataFrame, np.ndarray]] = None):   
        
        try:
            if self.loaded_model is None:
                
                # Calling class method to get the production model from aws s3 bucket
                # or we can call the ProductionModel/production_model.pkl from our project
                self.loaded_model = self.load_prod_model()
            
            return self.loaded_model.predict(X=X_data)   
            #return self.loaded_model.predict(X_data)
        
        except Exception as e:
            raise USvisaException(e, sys)
        
        

