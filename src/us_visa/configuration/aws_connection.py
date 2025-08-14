# boto3: Official AWS SDK for Python
import boto3
import os
from us_visa.constants import (AWS_SECRET_ACCESS_KEY_ENV_KEY, 
                               AWS_ACCESS_KEY_ID_ENV_KEY,
                               REGION_NAME)





class S3Client:
    
    """
    AWS S3 Client Singleton.

    This class manages a single reusable connection to AWS S3, using credentials from environment variables.
     
    It creates both:
        - boto3.resource (high-level S3 resource API)
        - boto3.client   (low-level S3 client API)

    Attributes:
        s3_client (boto3.client): Low-level S3 client.
        s3_resource (boto3.resource): High-level S3 resource.
    
    Example:
        >>> s3 = S3Client()
        >>> buckets = list(s3.s3_resource.buckets.all())
        >>> print(buckets)
    """

    # Global variable : Class-level shared AWS connections (singleton pattern)
    s3_client=None
    s3_resource = None
    
    def __init__(self, region_name=REGION_NAME):
        
        """
        Initialize AWS S3 connection using environment variables.

        Parameters : 
            (a) region_name (str): AWS region name. Defaults to `REGION_NAME` from constants.

        Raises:
            Exception: If AWS credentials are missing from environment variables.
        """

        if S3Client.s3_resource==None or S3Client.s3_client==None:
            
            # Load AWS credentials from environment variables
            __access_key_id=os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            __secret_access_key=os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)
            
            if __access_key_id is None:
                raise Exception(f"Environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY} is not not set.")
            
            if __secret_access_key is None:
                raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set.")

            # Setting the gobal variables using class name.
            # Create high-level S3 resource
            """
            Meaning:
                The 'boto3.resource('s3')' interface is a higher-level, object-oriented API for AWS S3.

            Why high-level?
                Instead of manually building and sending API requests, we interact with Python objects
                like 'Bucket' and 'Object' directly.
                
            When to use it:
                When we want simpler, object-like interactions with S3 : useful for quick 
                operations and clean code.
            """
            
            S3Client.s3_resource = boto3.resource('s3',
                                                  aws_access_key_id=__access_key_id,
                                                  aws_secret_access_key=__secret_access_key,
                                                  region_name=region_name
                                                 )
            
            # Create low-level S3 client
            """
            Meaning:
                The 'boto3.client('s3')' interface is a low-level, service API client for AWS S3.

            Why low-level?
                We call AWS operations directly as methods, passing in parameters exactly as 
                AWS's REST API expects.
                
            When to use it:
                When we need full control over request parameters or access to AWS features not 
                exposed in the high-level resource.
                
            """
            
            S3Client.s3_client = boto3.client('s3',
                                            aws_access_key_id=__access_key_id,
                                            aws_secret_access_key=__secret_access_key,
                                            region_name=region_name
                                            )
        
        # Assign instance attributes (point to shared connections)    
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client