from Restaurant_Rating_prediction.pipelines.prediction_pipeline import PredictPipeline, CustomData

# 🔹 **Create Input Data**
input_data = CustomData(
    online_order="Yes",
    book_table="Yes",
    votes=864,
    location="Residency Road",
    rest_type="Casual Dining",
    cuisines="Seafood, Mangalorean, North Indian, Chinese",
    approx_costfor_two_people=1400.0,
    listed_intype="Delivery"
)

# 🔹 **Convert to DataFrame**
input_df = input_data.get_data_as_df()

# 🔹 **Initialize Prediction Pipeline**
predict_pipeline = PredictPipeline()

# 🔹 **Make Prediction**
prediction = predict_pipeline.predict(input_df)

print(f"Predicted Rating: {prediction}")
