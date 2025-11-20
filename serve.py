from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# ----------------------
# Initialize app
# ----------------------
app = FastAPI(title="Airbnb Price Prediction API")

# ----------------------
# Root route (for testing API is running)
# ----------------------
@app.get("/")
def read_root():
    return {"message": "Airbnb Price Prediction API is running!"}

# ----------------------
# Load model and scaler
# ----------------------
model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# List of numeric columns used for scaling
num_cols = ['accommodates', 'bathrooms', 'bedrooms', 'beds',
            'latitude', 'longitude', 'number_of_reviews', 'review_scores_rating']

# ----------------------
# Input schema
# ----------------------
class AirbnbListing(BaseModel):
    accommodates: float
    bathrooms: float
    bedrooms: float
    beds: float
    latitude: float
    longitude: float
    number_of_reviews: float
    review_scores_rating: float
    cleaning_fee: int
    instant_bookable: int
    host_has_profile_pic: int
    host_identity_verified: int
    # Include categorical features as 0/1 flags if you want them in the API
    # For simplicity, optional; will be aligned automatically

# ----------------------
# Preprocessing function
# ----------------------
def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    # Ensure all expected features exist
    for col in model.feature_names_in_:
        if col not in data.columns:
            data[col] = 0  # add missing dummy columns as 0

    # Align column order
    data = data[model.feature_names_in_]

    # Scale numeric features safely
    data[num_cols] = scaler.transform(data[num_cols].to_numpy())

    return data

# ----------------------
# Prediction endpoint
# ----------------------
@app.post("/predict")
def predict(listing: AirbnbListing):
    # Convert input to DataFrame
    df = pd.DataFrame([listing.dict()])

    # Preprocess
    df_processed = preprocess_input(df)

    # Predict log price
    log_price_pred = model.predict(df_processed)[0]

    # Convert back to actual price
    price_pred = np.expm1(log_price_pred)

    return {
        "predicted_log_price": float(log_price_pred),
        "predicted_price": float(price_pred)
    }
