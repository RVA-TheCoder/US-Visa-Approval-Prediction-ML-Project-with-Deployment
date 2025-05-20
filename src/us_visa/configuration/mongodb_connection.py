import sys, os, pymongo, certifi
from us_visa.exception import USvisaException
from us_visa.logger import logging
#from us_visa.constants import DATABASE_NAME, MONGODB_URL_KEY
from us_visa.constants import DATABASE_NAME, MONGODB_URI_KEY
ca=certifi.where()


class MongoDBClient:
    
    """
    docs string to be added
    """
    # Global variable
    client=None
    
    def __init__(self, database_name=DATABASE_NAME):
        
        try :
            
            if MongoDBClient.client is None:
                # connection string to connect with our database
                #mongodb_url=os.getenv(MONGODB_URL_KEY)
                mongodb_uri=MONGODB_URI_KEY
                
                #if mongodb_url is None:
                if mongodb_uri is None:
                    #raise Exception(f"Environment Key : {MONGODB_URL_KEY} is not set.")
                    raise Exception(f"Environment Key : {MONGODB_URI_KEY} is not set.")
                
                # setting the global variable value
                # tlsCAFile: A file containing a single or a bundle of “certification authority” certificates, which are used to validate certificates passed from the other end of the connection.
                #MongoDBClient.client=pymongo.MongoClient(host=mongodb_url,tlsCAFile=ca)
                MongoDBClient.client=pymongo.MongoClient(host=mongodb_uri,tlsCAFile=ca)
                
                self.client=MongoDBClient.client  # this is optional
                self.database_name=database_name
                self.database=self.client[database_name]
                
                
                logging.info("MongoDB connection successful")
                
        except Exception as e :
            raise USvisaException(e,sys) from e
        
        


