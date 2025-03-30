import numpy as np
import pandas as pd
import sys
import os

from Restaurant_Rating_prediction.logger import logging
from Restaurant_Rating_prediction.exception import customexception

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Data_Ingestion_config:
    rawdata_path: str = os.path.join("Artifacts", "raw.csv")
    traindata_path: str = os.path.join("Artifacts", "train.csv")
    testdata_path: str = os.path.join("Artifacts", "test.csv")

class Data_Ingestion:
    def __init__(self):
        self.Data_Ingestion_config = Data_Ingestion_config()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Start")

        try:
            # ✅ Load dataset and reset index to ensure 'online_order' is a column
            data = pd.read_csv(Path(os.path.join("notebooks/data", "Zomato_cleaned2.csv")), index_col=0)
            data.reset_index(inplace=True)  # Ensures 'online_order' is a column
            logging.info("Data Read successfully")

            # ✅ Save raw data
            os.makedirs(os.path.dirname(self.Data_Ingestion_config.rawdata_path), exist_ok=True)
            data.to_csv(self.Data_Ingestion_config.rawdata_path, index=False)
            logging.info("Successfully saved raw data")

            # ✅ Train-test split
            logging.info("Performing train-test split")
            train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
            logging.info("Train-test split complete")

            # ✅ Save train and test data
            train_data.to_csv(self.Data_Ingestion_config.traindata_path, index=False)
            test_data.to_csv(self.Data_Ingestion_config.testdata_path, index=False)

            logging.info("Successfully saved train and test data")
            logging.info("Data Ingestion Complete")

            return train_data, test_data

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise customexception(e, sys)
