from flask import Blueprint, request, jsonify
import pandas as pd
import logging
import traceback
from flask_app.ml_utils import predict_market_price

# Setup logging
log_file = "notebooks/api.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create Blueprint
routes = Blueprint('routes', __name__)

@routes.route('/predict-market-price', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request body
        data = request.get_json() 
        logging.info(f"POST /predict-market-price - Raw JSON Data: {data}")

        # Convert JSON data to DataFrame with expected columns
        input_data = pd.DataFrame([{
    'Crop': data.get('crop'),
    'Region': data.get('region'), # Make sure it's a string like 'Low', 'Medium', etc.
    'Rainfall_mm': float(data.get('rainfall')),
    'Avg_Temperature_C': float(data.get('temperature')),
    'Production_Tonnes': float(data.get('production')),
    'MSP': float(data.get('msp')),
    'Export_Demand_Tonnes': float(data.get('Export_Demand_Tonnes')),
    'Prev_Year_Price': float(data.get('Prev_Year_Price'))
}])

        logging.info(f"POST /predict-market-price - Input DataFrame:\n{input_data}")

        # Call the prediction function from ml_utils.py
        prediction = predict_market_price(input_data)

        logging.info(f"POST /predict-market-price - Prediction Result: {prediction}")

        # Return the prediction as JSON
        return jsonify({
            "success": True,
            "prediction": prediction
        })

    except Exception as e:
        error_message = traceback.format_exc()
        logging.error(f"POST /predict-market-price - Error:\n{error_message}")

        # Return an error message as JSON
        return jsonify({
            "success": False,
            "error": "Prediction failed. Please check your input values."
        }), 500
