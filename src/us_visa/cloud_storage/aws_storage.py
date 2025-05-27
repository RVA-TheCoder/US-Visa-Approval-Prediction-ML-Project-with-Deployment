
from io import StringIO
from typing import Union,List
import os,sys
from pandas import DataFrame, read_csv
import pickle
import botocore
from pathlib import Path

from us_visa.logger import logging
from us_visa.exception import USvisaException

from mypy_boto3_s3.service_resource import Bucket
import boto3
from us_visa.configuration.aws_connection import S3Client
from botocore.exceptions import ClientError


# Creating Aws S3 utility methods using below class
class SimpleStorageService:

    def __init__(self):
        
        # local variable that exists only inside the __init__ method.
        s3_client = S3Client()
        
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client
    
        
    # Done    
    def get_bucket(self, bucket_name: str) -> Bucket:
        
        """
        Method Name :   get_bucket
        Description :   This method gets the bucket object based on the bucket_name

        Output      :   Bucket object is returned based on the bucket name
        On Failure  :   Write an exception log and then raise an exception

        """
        logging.info("Entered the get_bucket method of SimpleStorageService class inside src/us_visa/cloud_storage/aws_storage.py file")

        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            logging.info("Exited the get_bucket method of SimpleStorageService classinside src/us_visa/cloud_storage/aws_storage.py file")
            return bucket
        
        except Exception as e:
            raise USvisaException(e, sys) from e


    # Done
    def is_s3_key_path_available(self, bucket_name, s3_key)->bool:
        
        """
        Checks if a folder (key prefix) exists in an S3 bucket.
    
        Args:
            bucket_name (str): AWS S3 bucket name
            s3_key (str): Key prefix (folder path), e.g., 'models/'
    
        Returns:
            bool: True if the prefix exists, False otherwise
    
        Raises:
            Exception or USvisaException: If an error occurs
        """
        
        try:
            
            bucket = self.get_bucket(bucket_name)
            
            # Ensure s3_key ends with "/" (important when checking for folders)
            if not s3_key.endswith("/"):
                s3_key += "/"
    
            file_objects = list(bucket.objects.filter(Prefix=s3_key))

            return len(file_objects) > 0
            
        except Exception as e:
            raise USvisaException(e,sys)
        
    
    # Done
    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        
        """
        Creates a "folder" in an S3 bucket by putting a zero-byte object with a trailing slash in its key.
    
        Args:
            folder_name (str): Folder name (e.g., 'my_folder/')
            bucket_name (str): Name of the S3 bucket
    
        Raises:
            Exception or ClientError
        """
        
        logging.info("Entered the create_folder method of SimpleStorageService class inside src/us_visa/cloud_storage/aws_storage.py file")

        try:
            # Ensure folder_name ends with "/"
            if not folder_name.endswith("/"):
                folder_name += "/"
    
            # Try to load the folder object if it exists
            self.s3_resource.Object(bucket_name, folder_name).load()
            print(f"Folder {folder_name} already exists in the S3 bucket.")

        except ClientError as e:
            
            error_code = e.response["Error"]["Code"]
    
            if error_code == "404":
                # Folder doesn't exist, create it
                self.s3_client.put_object(Bucket=bucket_name, Key=folder_name)
                
            else:
                # Some other error occurred, raise it
                raise USvisaException(e,sys)
            
            logging.info("Exited the create_folder method of SimpleStorageService class inside src/us_visa/cloud_storage/aws_storage.py file")

    # Done
    # For checking if file exists or not
    def is_s3_file_exists(self, bucket_name: str, s3_key: str) -> bool:
        
        """
        Checks if a specific object (file) exists in the given S3 bucket.
    
        Args:
            bucket_name (str): Name of the S3 bucket.
            s3_key (str): Full S3 key (e.g., "ProductionModel/production_model.pkl")
    
        Returns:
            bool: True if the object exists, False otherwise.
        """
        
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            
            return True
            
        except botocore.exceptions.ClientError as e:
            
            if e.response["Error"]["Code"] == "404":
                return False
                
            # Some other error occurred, raise it
            raise USvisaException(e,sys)
     
    
    # Done
    def upload_file(self,
                    from_filename: str,
                    s3_key: str,
                    bucket_name: str,
                    remove: bool = False):
        
        """
        Uploads a file to an AWS S3 bucket.
    
        Args:
            from_filename (str): Local file path to upload.
            
            s3_key (str): The S3 object key (path in the bucket).
                          In AWS S3, the s3_key is the complete path including the file name — 
                          it uniquely identifies an object (file) in a bucket.
            
            bucket_name (str): Target S3 bucket name.
            remove (bool): Whether to delete the local file after upload. Defaults to False.

        """
        logging.info("Entered the upload_file method of SimpleStorageService class inside src/us_visa/cloud_storage/aws_storage.py file")

        try:
            logging.info(
                f"Uploading {from_filename} file to {s3_key} file in {bucket_name} bucket"
            )

            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, s3_key)
            print(f"Uploaded {from_filename} file to s3://{bucket_name}/{s3_key} path")

            logging.info(
                f"Uploaded {from_filename} file to {s3_key} file in {bucket_name} bucket"
            )

            if remove:
                os.remove(from_filename)
                print(f"Removed local file: {from_filename}")

                logging.info(f"Remove is set to {remove}, deleted the file")

            else:
                print(f"Local file retained: {from_filename}")
                logging.info(f"Remove is set to {remove}, not deleted the file")

            logging.info("Exited the upload_file method of SimpleStorageService class inside src/us_visa/cloud_storage/aws_storage.py file")
        
        except ClientError as e:
            print(f"ClientError during upload: {e}")
            
            raise USvisaException(e, sys) 
        
        except Exception as e:
            raise USvisaException(e, sys) from e
     
     
    # Done
    def get_s3_fileobjects( self, filename: str, bucket_name: str) -> Union[List[object], object]:
        
        """
        Method Name :   get_file_object
        Description :   The method retrieves file objects from an AWS S3 bucket based on a prefix (like a folder name or partial filename).

                        It returns either:
                            (a) A single object if only one match is found. 
                            (b) A list of objects if multiple matches are found.

        Output      :   list of objects or object is returned based on filename
        On Failure  :   Write an exception log and then raise an exception

        """
        
        logging.info("Entered the get_s3_fileobjects method of SimpleStorageService class inside src/us_visa/cloud_storage/aws_storage.py file")

        try:
            bucket = self.get_bucket(bucket_name)

            # This line filters all objects in the bucket whose key starts with 'filename'
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=filename)]

            func = lambda x: x[0] if len(x) == 1 else x

            file_objs = func(file_objects)
            logging.info("Exited the get_s3_fileobjects method of SimpleStorageService class inside src/us_visa/cloud_storage/aws_storage.py file")

            # Make it a list if not already
            if not isinstance(file_objs, list):
                file_objs = [file_objs]
            
            return file_objs

        except Exception as e:
            raise USvisaException(e, sys) from e


    # Done
    def download_s3_fileobject_as_file(self, object_summaries, destination_dir: Path):
    # def get_s3_object(self, object_summaries, destination_dir: Path):
        
        """
        Converts ObjectSummary(s) to actual S3 Object(s) and downloads the file(s) locally.
    
        Args:
            object_summaries (Union[List[s3.ObjectSummary], s3.ObjectSummary]): The ObjectSummary instance(s).
            destination_dir (Path): Local directory where files will be downloaded and saved.
        """
        logging.info("Entered the download_s3_fileobject_as_file method of SimpleStorageService class inside src/us_visa/cloud_storage/aws_storage.py file")
        # Ensure destination_dir is a directory and exists
        #destination_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(destination_dir, exist_ok=True)
     
        # Make it a list if not already
        if not isinstance(object_summaries, list):
            object_summaries = [object_summaries]

        for obj_summary in object_summaries:

            # Skip "folder" keys (those ending in "/")
            if obj_summary.key.endswith("/"):
                continue
    
            # Extract filename from key
            filename = os.path.basename(obj_summary.key)
            # Build full local path
            destination_path = destination_dir / filename
    
            # Download file
            s3_obj = self.s3_resource.Object(obj_summary.bucket_name, obj_summary.key)
            s3_obj.download_file(str(destination_path), )
            print(f"Downloaded: {destination_path}")

        logging.info("Exited the download_s3_fileobject_as_file method of SimpleStorageService class inside src/us_visa/cloud_storage/aws_storage.py file")
    
    # Done
    # def load_prod_model_from_local(self, filename local_s3_model_path:Path):
    def load_n_save_prod_model(self, 
                               s3_filename,
                               bucket_name, 
                               local_destination_dir:Path
                              ):
        """
        This method loads the production model downloaded from the S3 bucket and later saved on local system.
        
        Returns :  Skleanr S3 model object
        """
        try: 
            file_objects = self.get_s3_fileobjects(filename=s3_filename, 
                                                   bucket_name=bucket_name)
            print(file_objects)
    
            if len(file_objects) > 0 :
                
                self.download_s3_fileobject_as_file(object_summaries=file_objects, destination_dir=local_destination_dir)

                # Extract just the filename from the S3 key (e.g., 'model.pkl' from 'prod/model.pkl')
                model_filename = os.path.basename(s3_filename)
                # Now construct the actual file path to the downloaded model
                local_model_path = local_destination_dir / model_filename
            
                with open(local_model_path, mode='rb') as model_file:
                    s3_prod_model = pickle.load(model_file)
        
                return s3_prod_model

            else :
                print(f"{s3_filename} doesnot exist in S3 bucket")
                return None

        except Exception as e :
            raise USvisaException(e, sys)
        
        
    
         