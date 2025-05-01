# In this file, we're writing functions that we'll be used througout the project again & again.
import os, sys, dill, yaml 
import numpy as np
from pandas import DataFrame

from us_visa.exception import USvisaException
from us_visa.logger import logging





def read_yaml_file(filepath:str) ->dict:
    
    try :
        
        with open(filepath, "rb") as yaml_file:
            
            return yaml.safe_load(yaml_file)
        
        
    except Exception as e:  
        raise USvisaException(e, sys) from e
    


def write_yaml_file(filepath: str, content: object, replace: bool = False) -> None:
    """
    Writes content to a YAML file. Optionally prevents overwriting unless 'replace=True'.

    Parameters:
        filepath (str): Path to the YAML file.
        content (object): Python object to serialize as YAML.
        replace (bool): If False and file exists, raises an error. Default is False.

    Raises:
        USvisaException: If any error occurs during file operation.
    """
    try:
        
        if not replace and os.path.exists(filepath):
            raise USvisaException(f"File '{filepath}' already exists and replace=False", sys)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise USvisaException(e, sys) from e
    
    
    
def save_object(filepath:str, obj:object) -> None :
    
    
    """
        Serializes and saves the given object to a file.

        Parameters:
            filepath (str): The path where the object will be saved.
            obj (object): The object to be saved.
            
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
    
    '''
    This function loads a Python object (typically a model or serialized data) from a file
    using the dill library.
    
    It's similar to how we'd use 'pickle', but 'dill' is more powerful and can serialize more
    complex Python objects.
    
    '''
    
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
    This method saves numpy array data to a file.
    
    parameters :
    
       (a) filepath : str filepath to which the data is saved.
       (b) array : np.array data to save in above file
       
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
    This method loads a NumPy array from a .npy file.

    Parameters:
        filepath (str): Path to the .npy file from which data is loaded.

    Returns:
        np.ndarray: Loaded NumPy array.
    
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

    Parameters:
        df (pd.DataFrame): The DataFrame from which columns will be dropped.
        cols (list): List of column names to be dropped.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns dropped.
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













