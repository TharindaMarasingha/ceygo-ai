from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import math
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    year: int
    month: int

# Initialize the API
app = FastAPI(title="Ceygo Tourist Prediction API")

# CRITICAL FOR UI: Enable CORS so your frontend dashboard can fetch this data
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'sarima_monthly_model.pkl')

# Load the trained model into memory
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.get("/")
def home():
    return {"message": "Ceygo Prediction API is Live!"}

# NEW ENDPOINT: Specifically designed for your Dashboard Chart
@app.get("/api/prediction-chart")
def get_chart_data():
    try:
        # The last date our model was trained on
        last_training_date = pd.to_datetime('2023-12-01')
        
        # We want to predict exactly 12 months for a standard yearly chart
        months_ahead = 12
        forecast = model.predict(n_periods=months_ahead)
        
        # Generate the correct month/year labels for the X-axis
        future_dates = pd.date_range(
            start=last_training_date + pd.DateOffset(months=1), 
            periods=months_ahead, 
            freq='MS'
        )
        
        # Format the data exactly how frontend charting libraries want it
        chart_data = []
        for date, pred in zip(future_dates, forecast):
            chart_data.append({
                "month": date.strftime('%b %Y'),  # Formats to "Jan 2024", "Feb 2024", etc.
                "Arrivals": math.ceil(pred)       # The predicted number of tourists
            })
            
        return chart_data
    except Exception as e:
        return {"error": str(e)}
# NEW ENDPOINT: Specifically designed for your Custom ML Prediction Engine
@app.post("/api/predict-custom")
def predict_custom(req: PredictionRequest):
    try:
        last_training_date = pd.to_datetime('2023-12-01')
        target_date = pd.to_datetime(f"{req.year}-{req.month:02d}-01")
        
        # Calculate months difference for the specific prediction
        months_ahead = (target_date.year - last_training_date.year) * 12 + (target_date.month - last_training_date.month)
        
        if months_ahead <= 0:
            return {"error": "Target date must be in the future (after Dec 2023)."}
            
        # Predict up to December of the requested year to generate the chart data
        months_ahead_dec = (req.year - last_training_date.year) * 12 + (12 - last_training_date.month)
        
        forecast = model.predict(n_periods=months_ahead_dec)
        forecast_list = list(forecast)
        
        # The specific prediction
        target_prediction = forecast_list[months_ahead - 1]
        
        # Get the 12 months for that year (the last 12 elements of the forecast)
        year_forecast = forecast_list[-12:]
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        chart_data = []
        for i, pred in enumerate(year_forecast):
            chart_data.append({
                "month": months[i],
                "flights": math.ceil(pred),
                "historical": math.ceil(pred * (0.9 + 0.1 * (i % 3))) # slight mock variance for historical comparison
            })
        
        return {
            "prediction": math.ceil(target_prediction),
            "month": req.month,
            "year": req.year,
            "chartData": chart_data
        }
    except Exception as e:
        return {"error": str(e)}