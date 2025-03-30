import pandas as pd
import pickle

PREPROCESSOR_PATH = "Artifacts/preprocessor.pkl"

# Load preprocessor
with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

def preprocess_data(data):
    """Preprocess the input data using the saved preprocessor"""
    try:
        transformed_data = data.copy()
        
        # Apply transformations based on the preprocessor object
        for col, mapping in preprocessor.items():
            if col in transformed_data.columns:
                transformed_data[col] = transformed_data[col].map(mapping).fillna(0)
        
        return transformed_data
    except Exception as e:
        raise Exception(f"Error in preprocessing: {str(e)}")
