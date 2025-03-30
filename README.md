# Restaurant Rating Prediction API

This API predicts restaurant ratings based on various features such as online order availability, table booking options, location, restaurant type, cuisines, approximate cost for two people, and more.

## Project Structure
The project contains the following components:
- **Data Transformation**: Preprocesses the data and saves the preprocessor object as a pickle file.
- **Prediction Pipeline**: Uses the trained model to make predictions based on the provided input data.
- **API Endpoint**: Flask-based API to receive CSV files and return predictions.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ajaychaudhary2/RestaurantRatingPrediction.git
   cd RestaurantRatingPrediction
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the API
1. Start the Flask API:
   ```bash
   python app.py
   ```
2. The API will be running at:
   ```
   http://127.0.0.1:5000/predict
   ```

## Using Postman for Prediction
1. Open Postman and create a new request.
2. Set the method to **POST** and enter the URL:
   ```
   http://127.0.0.1:5000/predict
   ```
3. Go to the **Body** tab and select **form-data**.
4. Use the following key-value pair:
   - **Key**: `file` (select **File** type)
   - **Value**: Select the CSV file containing the input data.
5. Click **Send**.

### Example CSV Format
The input CSV file should have columns as follows:
```
online_order,book_table,votes,location,rest_type,cuisines,approx_costfor_two_people,listed_intype
Yes,No,500,Indiranagar,Quick Bites,Chinese,500,Dine-out
No,Yes,250,MG Road,Casual Dining,North Indian,1000,Delivery
```

### Response
The response will be a CSV file containing the predicted ratings, saved in the same directory with the filename `predictions.csv`.

## Troubleshooting
- If you encounter a `404 Not Found` error, ensure that the endpoint is correctly specified.
- Make sure to upload a valid CSV file as per the format shown above.

## License
This project is licensed under the MIT License.

