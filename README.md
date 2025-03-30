# Restaurant Rating Prediction API

This API predicts restaurant ratings based on various input features using a trained machine learning model. It also supports bulk predictions from a CSV file.

## Features
- Predicts restaurant ratings using individual inputs or a CSV file.
- Uses a pre-trained Random Forest Regressor model.
- Preprocessing is handled using a saved preprocessor object.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ajaychaudhary2/RestaurantRatingPrediction.git
   cd RestaurantRatingPrediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API:
   ```bash
   python app.py
   ```

## API Endpoints

### 1. Predict Single Rating
**URL:** `/predict`
**Method:** `POST`
**Content-Type:** `application/json`

#### Request Body Example:
```json
{
  "online_order": 1,
  "book_table": 0,
  "votes": 150,
  "location": "Indiranagar",
  "rest_type": "Casual Dining",
  "cuisines": "North Indian",
  "approx_costfor_two_people": 800,
  "listed_intype": "Buffet"
}
```

#### Response Example:
```json
{
  "predicted_rating": 4.2
}
```

### 2. Bulk Predict from CSV
**URL:** `/bulk_predict`
**Method:** `POST`
**Content-Type:** `multipart/form-data`

#### Request Example:
Using Postman, set the key as `file` and value as the CSV file.

#### Response Example:
A CSV file containing the predicted ratings will be returned for download.

## Usage with Postman
1. Open Postman and create a new request.
2. Select `POST` and enter the appropriate URL (e.g., `http://127.0.0.1:5000/predict`).
3. For single predictions, use `raw` and `JSON` for the body.
4. For bulk predictions, use `form-data` and upload the CSV file.
5. Click `Send` to get the prediction.

## Author
Ajay Chaudhary  
[LinkedIn](https://www.linkedin.com/in/ajay-chaudhary-02287a2ab/)

