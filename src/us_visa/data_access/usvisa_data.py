import pandas as pd
import sys
from typing import Optional
import numpy as np

from us_visa.configuration.mongodb_connection import MongoDBClient
from us_visa.constants import DATABASE_NAME
from us_visa.exception import USvisaException






class USvisaData:
    
    """
    Handles data extraction from MongoDB collections and conversion into Pandas DataFrames.

    This class:
      - Connects to the specified MongoDB database.
      - Fetches records from a given collection.
      - Cleans the DataFrame by removing MongoDB '_id' fields.
      - Replaces placeholder strings like "na" with actual NaN values.

    Example:
        data_access = USvisaData()
        df = data_access.export_collection_as_dataframe(collection_name="visa_records")
    """
    
    def __init__(self):
        
        """
        Initializes a MongoDB connection using the default DATABASE_NAME.
        """
        
        try :
            self.mongo_client=MongoDBClient(database_name=DATABASE_NAME)
            
        except Exception as e:
            raise USvisaException(e, sys) from e
        
        
    def export_collection_as_dataframe(self, collection_name:str,database_name:Optional[str]=None  ) -> pd.DataFrame:
        
        """
        Exports a MongoDB collection as a cleaned Pandas DataFrame.

        Parameters : 
            (a) collection_name (str): Name of the MongoDB collection to export.
            (b) database_name (Optional[str]): Name of the MongoDB database. 
                                               If None, uses the default DATABASE_NAME.

        Returns:
            pd.DataFrame: A DataFrame containing the collection's data, 
                          without the `_id` column, and with `"na"` replaced by NaN.

        Raises:
            USvisaException: If there is any error during data retrieval or processing.
        """
        
        try :
            # Select database and collection
            if database_name is None:
                collection=self.mongo_client.database[collection_name]
                
            else :
                collection=self.mongo_client[database_name][collection_name]
                
            # Convert MongoDB documents to DataFrame    
            df=pd.DataFrame( list( collection.find() ) )
            
            # Remove MongoDB's default `_id` field if present
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"], axis=1)
            
            # Replace placeholder "na" values with proper NaN for numeric compatibility   
            df.replace({"na":np.nan}, inplace=True)
            
            return df
        
        except Exception as e:
            
            raise USvisaException(e, sys) from e
        
        
                
            
    
    



