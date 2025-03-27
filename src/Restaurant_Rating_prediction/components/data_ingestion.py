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
    rawdata_path:str=os.path.join("Artifacts","raw.csv")
    traindata_path:str=os.path.join("Artifacts","train.csv")
    testdata_path:str=os.path.join("Artifacts","test.csv")
    
    
class Data_Ingestion:
    def __init__(self):
        self.Data_Ingestion_config=Data_Ingestion_config()
        
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Start")
        
        try:
        
        
          data =  pd.read_csv(Path(os.path.join("notebooks/data","Zomato_cleaned.csv")),index_col=0)
          logging.info("Data Read succesfully")
        
          os.makedirs(os.path.dirname(self.Data_Ingestion_config.rawdata_path), exist_ok=True)
          data.to_csv(self.Data_Ingestion_config.rawdata_path,index=False)
          logging.info("Succesfully saved raw data")
        
        
          logging.info("Performing train test split ")
          train_data,test_data=train_test_split(data,test_size=.25)
          logging.info("train test split complete")
        
          train_data.to_csv(self.Data_Ingestion_config.traindata_path,index=False)
          test_data.to_csv(self.Data_Ingestion_config.testdata_path,index=False)
          
          logging.info("Succesfully save the Test and Train data")
          logging.info("Data Ingestion Complete")
          
          return train_data , test_data
        
     
        except Exception as e:
              logging.error("Error come in the data ingestion")
              raise customexception(e,sys)