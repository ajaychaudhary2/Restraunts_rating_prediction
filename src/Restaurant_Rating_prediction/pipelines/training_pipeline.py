from Restaurant_Rating_prediction.logger import logging
from Restaurant_Rating_prediction.exception import customexception
from Restaurant_Rating_prediction.components.data_ingestion import Data_Ingestion
from Restaurant_Rating_prediction.components.data_transformation import DataTransformation
from Restaurant_Rating_prediction.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        # Step 1: Data Ingestion
        obj = Data_Ingestion()
        train_df, test_df = obj.initiate_data_ingestion()
        
        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_df, test_df)

        
        
        # Step 3: Model Training
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)  

        print("Training pipeline executed successfully.")

    except customexception as e:
        logging.error(f"Custom Exception: {str(e)}")
