# In this file, we're writing functions that we'll be used througout the project again & again.
import os, sys, dill, yaml 
import numpy as np
from pandas import DataFrame

from us_visa.exception import USvisaException
from us_visa.logger import logging



"""
Utility functions for the US Visa Approval Prediction project.

This module provides reusable helper functions for:
    - Reading and writing YAML files
    - Saving and loading serialized Python objects
    - Handling NumPy array persistence
    - Performing common DataFrame transformations

These utilities help avoid code duplication and ensure consistency across the pipeline.
"""



def read_yaml_file(filepath:str) ->dict:
    
    """
    Read and parse a YAML file.

    Parameters : 
        (a) filepath (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content as a Python dictionary.

    Raises:
        USvisaException: If the file cannot be read or parsed.
    """
    
    try :
        
        with open(filepath, "rb") as yaml_file:
            
            return yaml.safe_load(yaml_file)
        
        
    except Exception as e:  
        raise USvisaException(e, sys) from e
    


def write_yaml_file(filepath: str, content: object, replace: bool = True) -> None:
    
    """
    Writes content to a YAML file. Optionally prevents overwriting unless 'replace=True'.

    Parameters:
        (a) filepath (str): Path to the YAML file.
        (b) content (object): Python object to serialize as YAML.
        (c) replace (bool, optional): Whether to overwrite the file if it exists. Defaults to True.

    Raises:
        USvisaException: If any error occurs during file operation.
    """
    try:
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise USvisaException(e, sys) from e
    
    
    
def save_object(filepath:str, obj:object) -> None :
    
    """
    Serialize and save a Python object to disk using dill.

    Parameters : 
        (a) filepath (str): Path where the object will be saved.
        (b) obj (object): Python object to serialize.

    Raises:
        USvisaException: If saving fails.
    """
    
    logging.info("Entered the save_object method of utils/main_utils.py")
    
    try:
        
        if not filepath:
            raise ValueError("File path must not be empty.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "wb") as file:
            
            dill.dump(obj, file)
            
        logging.info("Exited the save_object method of utils/main_utils.py")
        
    except Exception as e :
        
        raise USvisaException(e, sys) from e
    
    

def load_object(filepath:str) -> object:
    
    """
    Load a serialized Python object from disk using dill.

    Parameters :
        (a) filepath (str): Path to the file containing the serialized object.

    Returns:
        object: Deserialized Python object.

    Raises:
        USvisaException: If loading fails.
    """
    
    logging.info("Entered the load_object method of utils/main_utils.py") 
    
    try :
        
        with open(filepath, "rb") as file:
            
            # dill.load() : to deserialize the object from the binary file.
            # Equivalent to pickle.load(), but dill can handle more advanced Python objects (like lambdas, closures, etc.).
            
            obj=dill.load(file)
            
        logging.info("Exited the load_object method of utils/main_utils.py")
        
        return obj
    
    except Exception as e:
        raise USvisaException(e, sys) from e



def save_numpy_array_data(filepath:str, array:np.ndarray):
    
    """
    Save a NumPy array to disk in .npy format.

    Parameters : 
        (a) filepath (str): Destination file path.
        (b) array (np.ndarray): NumPy array to save.

    Raises:
        USvisaException: If saving fails.
    """
    
    try:
        
        if not filepath:
            raise ValueError("File path must not be empty.")
        
        dir_path=os.path.dirname(filepath)
        
        os.makedirs(dir_path, exist_ok=True)
        
        # Save an array to a binary file in NumPy .npy format.
        np.save(filepath, array)
            
            
    except Exception as e:
        raise USvisaException(e, sys)
    
    

def load_numpy_array_data(filepath:str) -> np.ndarray:
    
    """
    Load a NumPy array from a .npy file.

    Parameters : 
        (a) filepath (str): Path to the .npy file.

    Returns:
        np.ndarray: Loaded NumPy array.

    Raises:
        USvisaException: If loading fails or file does not exist.
    """
    
    try :
        
        if not os.path.exists(filepath):
            raise USvisaException(f"File '{filepath}' not found.", sys)
        
        # np.load : open the file in "rb" mode automatically
        return np.load(filepath)
  
    except Exception as e :
        raise USvisaException(e, sys) from e
    
 
    
def drop_columns(df: DataFrame, cols: list) -> DataFrame:

    """
    Drop specified columns from a pandas DataFrame.

    Parameters :
        (a) df (DataFrame): Input DataFrame.
        (b) cols (list): List of column names to drop.

    Returns:
        DataFrame: New DataFrame without the specified columns.

    Raises:
        ValueError: If any specified column is not found in the DataFrame.
        USvisaException: If the operation fails.
    """
    
    logging.info("Entered drop_columns method for dropping columns from a DataFrame of utils/main_utils.py")

    try:
        
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        #Drop the specified columns
        df = df.drop(columns=cols, axis=1)

        logging.info("Exited the drop_columns method of utils/main_utils.py")
        
        return df
    
    except Exception as e:
        raise USvisaException(e, sys) from e    










