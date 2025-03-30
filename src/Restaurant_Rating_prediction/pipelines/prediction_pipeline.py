import os
import sys
import pandas as pd
import numpy as np
from Restaurant_Rating_prediction.logger import logging
from Restaurant_Rating_prediction.exception import customexception
from Restaurant_Rating_prediction.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        try:
            self.model_path = os.path.join("Artifacts", "model.pkl")
            self.preprocessor_path = os.path.join("Artifacts", "preprocessor.pkl")

            logging.info("Loading preprocessor and model...")
            self.preprocessor = load_object(self.preprocessor_path)  # ✅ Preprocessing Pipeline
            self.model = load_object(self.model_path)  # ✅ Trained Model

            logging.info("Preprocessor and model loaded successfully!")
        except Exception as e:
            logging.error("Error loading model or preprocessor")
            raise customexception(e, sys)

    def predict(self, input_df):
        try:
            logging.info(f"Raw Input Data: {input_df}")

            # ✅ **Apply Preprocessing Pipeline**
            input_transformed = self.preprocessor.transform(input_df)

            # ✅ **Make Prediction**
            prediction = self.model.predict(input_transformed)

            logging.info(f"Prediction: {prediction[0]}")
            return prediction[0]
        
        except Exception as e:
            logging.error("Error during prediction")
            raise customexception(e, sys)

# ✅ **CustomData Class for Handling Input Data**
class CustomData:
    def __init__(self, online_order, book_table, votes, location, rest_type, cuisines, approx_costfor_two_people, listed_intype):
        self.online_order = online_order
        self.book_table = book_table
        self.votes = votes
        self.location = location
        self.rest_type = rest_type
        self.cuisines = cuisines
        self.approx_costfor_two_people = approx_costfor_two_people
        self.listed_intype = listed_intype

    def get_data_as_df(self):
        """Converts input data into a Pandas DataFrame"""
        try:
            logging.info("Converting input data to DataFrame")
            data_dict = {
                "online_order": [self.online_order],
                "book_table": [self.book_table],
                "votes": [self.votes],
                "location": [self.location],
                "rest_type": [self.rest_type],
                "cuisines": [self.cuisines],
                "approx_costfor_two_people": [self.approx_costfor_two_people],
                "listed_intype": [self.listed_intype]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            logging.error("Error converting input data to DataFrame")
            raise customexception(e, sys)
