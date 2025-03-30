import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Restaurant_Rating_prediction.exception import customexception
from Restaurant_Rating_prediction.logger import logging
from Restaurant_Rating_prediction.utils.utils import save_object

class DataTransformationConfig:
    """Configuration class for storing file paths."""
    preprocessor_obj_file_path = os.path.join("Artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.datatransformation_config = DataTransformationConfig()

    def get_data_transformation(self, train_df, test_df):
        """Performs data preprocessing including encoding."""
        try:
            logging.info("Starting data transformation")

            # ✅ Required columns for transformation
            required_cols = [
                'online_order', 'book_table', 'location', 'rest_type', 
                'cuisines', 'listed_intype', 'votes', 'approx_costfor_two_people'
            ]
            for col in required_cols:
                if col not in train_df.columns or col not in test_df.columns:
                    raise KeyError(f"Column '{col}' is missing in the dataset.")

            preprocessor = {}

            # ✅ Label Encoding for Binary Features
            le = LabelEncoder()
            for col in ['online_order', 'book_table']:
                logging.info(f"Encoding binary feature: {col}")
                train_df[col] = le.fit_transform(train_df[col])
                test_df[col] = le.transform(test_df[col])
                preprocessor[col] = dict(zip(le.classes_, le.transform(le.classes_)))

            # ✅ Frequency Encoding for Categorical Features
            for col in ['location', 'rest_type', 'cuisines', 'listed_intype']:
                logging.info(f"Encoding categorical feature: {col}")
                freq_map = train_df[col].value_counts(normalize=True).to_dict()
                train_df[col] = train_df[col].map(freq_map)
                test_df[col] = test_df[col].map(freq_map).fillna(0)
                preprocessor[col] = freq_map

            logging.info("Data transformation completed successfully")
            return train_df, test_df, preprocessor

        except KeyError as e:
            logging.error(f"Missing Column Error: {str(e)}")
            raise customexception(e, sys)
        except Exception as e:
            logging.error("Exception occurred during data transformation")
            raise customexception(e, sys)

    def initiate_data_transformation(self, train_df, test_df):
        """Applies preprocessing and saves the preprocessor object."""
        try:
            train_df, test_df, preprocessor = self.get_data_transformation(train_df, test_df)

            # ✅ Check Preprocessor Object
            if not preprocessor:
                logging.error("Preprocessor object is empty. Aborting save operation.")
                raise customexception("Preprocessor object is empty", sys)

            # ✅ Log Preprocessor Object Contents
            logging.info(f"Preprocessor object content: {preprocessor}")

            # ✅ Save Preprocessor Object
            save_object(
                file_path=self.datatransformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Preprocessor object saved successfully")
            return train_df, test_df

        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise customexception(e, sys)
