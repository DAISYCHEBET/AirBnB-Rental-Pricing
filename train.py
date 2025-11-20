# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ----------------------
# Functions
# ----------------------

def preprocess_data(df, num_fill_median, binary_cols, categorical_cols, fit_scaler=True, scaler=None):
    # Drop unnecessary columns
    drop_cols = ["id", "name", "description", "thumbnail_url",
                 "first_review", "last_review", "host_since", "zipcode", "amenities"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Fill numeric missing values
    for col in num_fill_median:
        df[col] = df[col].fillna(df[col].median())

    # Binary columns
    for col in binary_cols:
        df[col] = df[col].fillna(False)
    df[binary_cols] = df[binary_cols].replace({'t':1, 'f':0, True:1, False:0}).astype(int)

    # Categorical columns
    df['neighbourhood'] = df['neighbourhood'].fillna("Unknown")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Separate features and target
    X = df.drop('log_price', axis=1)
    y = df['log_price']

    # Scale numeric columns
    if fit_scaler:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols].to_numpy())
    else:
        X[num_cols] = scaler.transform(X[num_cols].to_numpy())

    return X, y, scaler

# ----------------------
# Parameters
# ----------------------
num_fill_median = ["bathrooms", "bedrooms", "beds", "review_scores_rating"]
binary_cols = ["cleaning_fee", "instant_bookable", "host_has_profile_pic", "host_identity_verified"]
categorical_cols = ['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city', 'neighbourhood']
num_cols = ['accommodates', 'bathrooms', 'bedrooms', 'beds',
            'latitude', 'longitude', 'number_of_reviews', 'review_scores_rating']

# ----------------------
# Load dataset
# ----------------------
df = pd.read_csv("Airbnb_Data.csv")

# ----------------------
# Preprocess
# ----------------------
X, y, scaler = preprocess_data(df, num_fill_median, binary_cols, categorical_cols, fit_scaler=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------
# Train Random Forest
# ----------------------
rf_model = RandomForestRegressor(
    n_estimators=200,
    min_samples_split=10,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Tuned Random Forest RMSE: {rmse:.4f}")
print(f"Tuned Random Forest R²: {r2:.4f}")

# Save model and scaler
joblib.dump(rf_model, "best_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved.")
