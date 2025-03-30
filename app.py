from flask import Flask, request, jsonify
import pandas as pd
import os
import pickle
import traceback
from Restaurant_Rating_prediction.utils.preprocess import preprocess_data

app = Flask(__name__)

MODEL_PATH = "Artifacts/model.pkl"
UPLOAD_DIR = "uploads/"
OUTPUT_DIR = "predictions/"

# Load the model
model = pickle.load(open(MODEL_PATH, "rb"))

@app.route('/')
def home():
    return "Welcome to the CSV Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Save uploaded CSV
        file = request.files['file']
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(file_path)

        # Load CSV
        data = pd.read_csv(file_path)

        # Preprocess data
       
        transformed_data = preprocess_data(data)

        # Predict
        predictions = model.predict(transformed_data)

        # Save predictions
        data['predicted_rating'] = predictions
        output_path = os.path.join(OUTPUT_DIR, "predictions_" + file.filename)
        data.to_csv(output_path, index=False)

        return jsonify({"message": "Prediction complete", "output_file": output_path})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    app.run(debug=True)
