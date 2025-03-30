import os
import sys
import pickle
from Restaurant_Rating_prediction.logger import logging
from Restaurant_Rating_prediction.exception import customexception

def save_object(file_path, obj):
    """
    Saves a Python object (model, preprocessor, etc.) to a specified file path using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Successfully saved object of type {type(obj)} at {file_path}")

    except Exception as e:
        logging.error(f"Error saving object at {file_path}")
        raise customexception(e, sys)
    
def load_object(file_path):
    """
    Loads a Python object (model, preprocessor, etc.) from a specified file path.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
            logging.info(f"Loaded object of type {type(obj)} from {file_path}")
            return obj
    except Exception as e:
        logging.error(f"Exception occurred while loading object from {file_path}")
        raise customexception(e, sys)
