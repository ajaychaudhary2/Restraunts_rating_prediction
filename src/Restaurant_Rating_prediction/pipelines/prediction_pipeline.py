import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Restaurant_Rating_prediction.exception import customexception
from Restaurant_Rating_prediction.logger import logging
from Restaurant_Rating_prediction.utils.utils import load_object


# ✅ CustomData Class for Handling Input Data
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


# ✅ Prediction Pipeline Class
class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("Artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("Artifacts", "preprocessor.pkl")

    def predict(self, features):
        """Loads model and preprocessor, then performs prediction"""
        try:
            logging.info("Loading the preprocessor and model for prediction")

            # Load preprocessor and model
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            # Transform features using the preprocessor
            transformed_features = features.copy()

            # Apply encoding transformations
            for col, enc_map in preprocessor.items():
                if col in transformed_features.columns:
                    transformed_features[col] = transformed_features[col].map(enc_map).fillna(0)

            logging.info("Prediction started")
            prediction = model.predict(transformed_features)
            logging.info("Prediction completed")

            return prediction

        except Exception as e:
            logging.error("Error occurred during prediction")
            raise customexception(e, sys)



if __name__ == "__main__":
    try:
        # Example input data
        input_data = CustomData(
            online_order="No",
            book_table="Yes",
            votes=1192,
            location="JP Nagar",
            rest_type="Italian, Continental",
            cuisines="Casual Dining, Pub",
            approx_costfor_two_people=1200.0,
            listed_intype="Delivery"
        )

        # Convert input data to DataFrame
        data_df = input_data.get_data_as_df()

        # Create Prediction Pipeline and make a prediction
        pipeline = PredictPipeline()
        prediction = pipeline.predict(data_df)

        print(f"Predicted Rating: {prediction[0]}")
    except Exception as e:
        print(f"Error in main prediction: {str(e)}")
