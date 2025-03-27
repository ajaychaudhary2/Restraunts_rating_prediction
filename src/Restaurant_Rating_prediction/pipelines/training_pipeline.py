from Restaurant_Rating_prediction.logger import logging
from Restaurant_Rating_prediction.exception import customexception

from Restaurant_Rating_prediction.components.data_ingestion import Data_Ingestion


if __name__=="__main__":
    
    try:
        obj=Data_Ingestion()
        train_df, test_df=obj.initiate_data_ingestion()
    
    
    
    except customexception as e:
        logging.error(f"Custom Exception: {str(e)}")