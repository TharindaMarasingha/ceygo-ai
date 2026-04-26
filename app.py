from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import math

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

# Load the trained model into memory
with open('sarima_monthly_model.pkl', 'rb') as file:
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