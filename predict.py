import pandas as pd
import numpy as np
import joblib

# ----------------------
# Functions
# ----------------------

def preprocess_new_data(new_data, num_fill_median, binary_cols, categorical_cols, scaler, feature_columns):
    # Fill numeric missing values
    for col in num_fill_median:
        new_data[col] = new_data[col].fillna(new_data[col].median())

    # Binary columns
    new_data[binary_cols] = new_data[binary_cols].replace({'t':1, 'f':0, True:1, False:0}).astype(int)

    # Categorical columns
    new_data['neighbourhood'] = new_data['neighbourhood'].fillna("Unknown")
    new_data = pd.get_dummies(new_data, columns=categorical_cols, drop_first=True)

    # Align with training features
    for col in feature_columns:
        if col not in new_data.columns:
            new_data[col] = 0
    new_data = new_data[feature_columns]

    # Scale numeric columns
    new_data[num_cols] = scaler.transform(new_data[num_cols].to_numpy())

    return new_data

# ----------------------
# Parameters
# ----------------------
num_fill_median = ["bathrooms", "bedrooms", "beds", "review_scores_rating"]
binary_cols = ["cleaning_fee", "instant_bookable", "host_has_profile_pic", "host_identity_verified"]
categorical_cols = ['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city', 'neighbourhood']
num_cols = ['accommodates', 'bathrooms', 'bedrooms', 'beds',
            'latitude', 'longitude', 'number_of_reviews', 'review_scores_rating']

# ----------------------
# Load model and scaler
# ----------------------
model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------
# Load new data
# ----------------------
new_data = pd.read_csv("new_airbnb_listings.csv")

# ----------------------
# Preprocess new data
# ----------------------
new_data_processed = preprocess_new_data(
    new_data, num_fill_median, binary_cols, categorical_cols, scaler, model.feature_names_in_
)

# ----------------------
# Predict
# ----------------------
preds = model.predict(new_data_processed)
new_data['predicted_log_price'] = preds
new_data['predicted_price'] = np.expm1(preds)  # optional: convert back from log price

# Save predictions
new_data.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")
