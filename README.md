🏡 Airbnb Rental Price Prediction

ML Zoomcamp Midterm Project – 2025

This project predicts Airbnb rental prices based on listing features.
It includes data cleaning, model training, evaluation, and a fully packaged FastAPI web service deployed using Docker.

📌 Project Overview

Airbnb hosts often struggle to price their listings accurately.
This project solves that problem using a machine learning model that estimates the expected rental price of a listing based on:

Property type

Room type

Location

Amenities

Host attributes

And other listing features

The final model is served as an API using FastAPI, with deployment handled via Docker.

📂 Repository Structure
AirBnB-Rental-Pricing/
│
├── data/                     # Dataset 
│
├── train.py                  # Script to train the model
├── predict.py                # Helper script to load the model and make predictions
├── serve.py                  # FastAPI app for serving predictions
│
├── model.pkl                 # Saved trained model
├── requirements.txt          # Project dependencies
├── Dockerfile                # Docker configuration
│
└── README.md                 # This file






🧹 1. Data Preparation & EDA

Loaded and cleaned the dataset

Handled missing values

Encoded categorical variables

Scaled numerical features

Performed exploratory data analysis

Removed outliers and transformed skewed variables





🤖 2. Model Training

Multiple models were trained and compared:

Linear Regression

Random Forest

XGBoost (final model)

The final model was selected based on lowest RMSE and best generalization.

Training is handled in:

train.py


After tuning, the final model is saved as:

model.pkl

⚙️ 3. Prediction Logic

All inference logic is in:

predict.py


The script:

Loads the saved model

Preprocesses incoming data

Returns the predicted rental price

🌐 4. API Service (FastAPI)

The API is defined in:

serve.py

Endpoints:
GET /

Returns:

{"message": "Airbnb Price Prediction API is running!"}

POST /predict

Example request:

{
  "property_type": "Apartment",
  "room_type": "Entire home/apt",
  "bathrooms": 1,
  "bedrooms": 2,
  "accommodates": 4,
  "cleaning_fee": 1,
  "city": "Nairobi",
  ...
}


Example response:

{
  "predicted_price": 68.4
}

🐳 5. Docker Deployment
Build the image
docker build -t airbnb-price-api .

Run the container
docker run -d -p 8000:8000 airbnb-price-api

Test the API

Open in your browser:

http://127.0.0.1:8000

http://127.0.0.1:8000/docs





🧪 6. Testing the API

You can also send a request using curl:

curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d @example.json


Or use the Swagger UI at:

http://127.0.0.1:8000/docs




🔁 Reproducibility

To reproduce the project:

git clone <repo_url>
cd AirBnB-Rental-Pricing
pip install -r requirements.txt
python serve.py


Or run via Docker (recommended):

docker build -t airbnb-price-api .
docker run -p 8000:8000 airbnb-price-api

📝 Final Notes

This project was completed as part of the ML Zoomcamp Midterm, focusing on:

Proper ML workflow

Reproducibility

Model deployment

Dockerization

API design

Author

Daisy
Kenya 🇰🇪