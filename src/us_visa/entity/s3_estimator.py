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
    Handles saving, loading, and retrieving the US Visa prediction model to and from AWS S3 storage.
    

    Responsibilities:
        - Upload trained models to a specific S3 bucket path.
        - Check if a production model exists in S3.
        - Load the production model from S3 into the local environment.
        - (Optional) Run predictions using the loaded model.
    """

    def __init__(self, bucket_name, s3_prod_model_path=f"{MODEL_PUSHER_S3_KEY}/{S3_PRODUCTION_MODEL_NAME}"):
        
        """
        Initializes the S3 Estimator with the bucket name and production model path.

        Parameters : 
            (a) bucket_name (str): Name of the S3 bucket where the model is stored.
            (b) s3_prod_model_path (str): S3 key path of the production model.
        """
        
        self.bucket_name = bucket_name
        self.s3_model_path = s3_prod_model_path   # need to work on this f"{MODEL_PUSHER_S3_KEY}/{S3_PRODUCTION_MODEL_NAME}"
        self.s3 = SimpleStorageService()
        
        #self.loaded_model:USvisaModel=None
        self.loaded_model=None


    def is_s3_model_present(self,):
        
        """
        Checks whether the production model file exists in the S3 bucket.

        Returns:
            bool: True if model file exists in S3, False otherwise.
        """
        
        """
        we can use below methods :
        
        (i) is_s3_file_exists(bucket_name=MODEL_BUCKET_NAME, 
                           s3_key=f"{MODEL_PUSHER_S3_KEY}/production_model.pkl") : bool
        (ii) get_s3_file_objects(filename=f"{MODEL_PUSHER_S3_KEY}/my_production_model.pkl", 
                                  bucket_name=MODEL_BUCKET_NAME) 
        (iii) self.loaded_model= download_s3_fileobject_as_file(object_summaries=file_objects1, destination_dir=Path("ProductionModel"))
        
        
        """
        
        try:
            
            # returns True or False 
            return self.s3.is_s3_file_exists(bucket_name=self.bucket_name, 
                                             s3_key=self.s3_model_path)
        
        except USvisaException as e: 
            print(e)
            
            return False


  
    def load_prod_model(self, ):
        
        """
        Downloads the production model from S3 into the local directory.

        Returns:
            Path: Local path to the downloaded model file.
        """
        
        """
        we can use below methods :
            (i) is_model_present(self, model_path) : bool 
            (ii) get_s3_file_objects(filename=f"{MODEL_PUSHER_S3_KEY}/my_production_model.pkl", 
                                    bucket_name=MODEL_BUCKET_NAME) 
            (iii) self.loaded_model= download_s3_fileobject_as_file(object_summaries=file_objects1, destination_dir=Path("ProductionModel"))
        
        """
        
        try:
            return self.s3.load_n_save_prod_model(
                s3_filename=self.s3_model_path,
                bucket_name=self.bucket_name,
                local_destination_dir=Path(LOCAL_PRODUCTION_MODEL_DIR)
            )
        except Exception as e:
            raise USvisaException(e, sys)
        


    # We already defined the methods get_s3_file_objects and download_s3_fileobject_as_file
    # we use above methods to save the model in our local system
    def upload_model_to_s3(self, from_filepath, remove:bool=False)->None:
        
        """
        Uploads a model file from local storage to the S3 production model path.

        Parameters : 
            (a) from_filepath (str | Path): Path to the local model file.
            (b) remove (bool): If True, deletes the local file after upload.
        """
        
        try:
            self.s3.upload_file(
                from_filename=from_filepath,
                bucket_name=self.bucket_name,
                s3_key=self.s3_model_path,
                remove=remove
            )
        except Exception as e:
            raise USvisaException(e, sys)


        
        

