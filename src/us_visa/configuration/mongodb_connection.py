import sys, os, pymongo, certifi
from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.constants import DATABASE_NAME

from urllib.parse import quote_plus
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Path to CA certificates for secure MongoDB connection
ca=certifi.where()



class MongoDBClient:
    
    """
    MongoDB Client for connecting to the US Visa database.

    This class establishes a secure connection to MongoDB using either:
        1. A pre-defined connection string from the environment variable 'MONGODB_URI_KEY'.
        2. A fallback username/password-based URI (for local development).

    Attributes:
        client (pymongo.MongoClient): Shared MongoDB connection instance
        database_name (str): Name of the target database.
        database (pymongo.database.Database): The connected database instance.

    Example:
        >>> mongo_client = MongoDBClient()
        >>> collection = mongo_client.database["visa_applications"]
        >>> collection.find_one()
    """
    
    # Global variable : Class-level shared MongoDB client (singleton)
    client=None
    
    def __init__(self, database_name=DATABASE_NAME):
        
        """
        Initialize the MongoDB connection.

        Parameters : 
            (a) database_name (str): Name of the MongoDB database to connect to.
                                     Defaults to 'DATABASE_NAME' from constants.

        Raises:
            USvisaException: If connection fails or environment variables are missing.
        """
        
        try :
            
            if MongoDBClient.client is None:
                
                # Try CI/CD environment variable first
                mongodb_uri = os.getenv("MONGODB_URI_KEY")
                
                # Fallback to local development : Build URI from local username & password
                if not mongodb_uri:
                    
                    username = os.getenv("MONGO_USERNAME")
                    password = quote_plus(os.getenv("MONGO_PASSWORD"))
                    mongodb_uri = f"mongodb+srv://{username}:{password}@us-visa-ml-cluster.q9tznwq.mongodb.net/?retryWrites=true&w=majority&appName=US-VISA-ML-Cluster"
                
                
                if mongodb_uri is None:
                    
                    raise Exception(f"Environment Key : {mongodb_uri} environment variables not set.")
                
                # Establish secure MongoDB connection with TLS
                # tlsCAFile: A file containing a single or a bundle of "certification authority" certificates, which are used to validate certificates passed from the other end of the connection.
                # setting the global variable value
                MongoDBClient.client=pymongo.MongoClient(host=mongodb_uri, tlsCAFile=ca)
                
                # Assign instance attributes
                self.client=MongoDBClient.client  
                self.database_name=database_name
                self.database=self.client[database_name]
                
                
                logging.info("MongoDB connection successful")
                
        except Exception as e :
            raise USvisaException(e,sys) from e
        
        


