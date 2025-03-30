import os
import sys
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from Restaurant_Rating_prediction.logger import logging
from Restaurant_Rating_prediction.exception import customexception
from Restaurant_Rating_prediction.utils.utils import save_object

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Artifacts", "model.pkl")
    preprocessor_file_path = os.path.join("Artifacts", "preprocessor.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Loading preprocessor object")
            
            # Load the preprocessor.pkl
            with open(self.model_trainer_config.preprocessor_file_path, "rb") as f:
                preprocessor = pickle.load(f)

            # ✅ Ensure train_arr is a NumPy array before indexing
            if isinstance(train_arr, np.ndarray):
                print("✅ train_arr is a NumPy array")
            else:
                print("⚠️ train_arr is NOT a NumPy array! Converting it now.")
                train_arr = train_arr.to_numpy()
                test_arr = test_arr.to_numpy()

            # Debugging train_arr before slicing
            print("Train array shape:", train_arr.shape)
            print("Test array shape:", test_arr.shape)
            print("First row of train array:", train_arr[0, :])  # ✅ FIXED INDEXING

            # Extracting Features & Target
            X_train = train_arr[:, :-1]  # ✅ All columns except last (features)
            y_train = train_arr[:, -1]   # ✅ Last column (target)

            X_test = test_arr[:, :-1]    # ✅ Features
            y_test = test_arr[:, -1]     # ✅ Target

            # Debugging extracted X_train & y_train
            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)
            print("First row X_train:", X_train[0])
            print("First value y_train:", y_train[0])

            logging.info("Starting model training with RandomForestRegressor")

            # Initialize and Train Model
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Model Performance
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

            # Logging Model Performance
            logging.info(f"Train R² Score: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
            logging.info(f"Test R² Score: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

            # Save Model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            logging.info("Model saved successfully as model.pkl")

        except Exception as e:
            logging.error("Exception occurred during Model Training")
            raise customexception(e, sys)
