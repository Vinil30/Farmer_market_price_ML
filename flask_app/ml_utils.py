import pickle
import pandas as pd
import logging
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))  # This finds the folder path of ml_utils.py

model_path = os.path.join(base_dir, "models", "random_forest_model.pkl")
scaler_path = os.path.join(base_dir, "models", "standard_scaler.pkl")
encoder_path = os.path.join(base_dir, "models", "onehot_encoder.pkl")


with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)

def predict_market_price(input_df):
    try:
        logging.info(f"Received input for prediction:\n{input_df}")

        # Define your features
        numerical_features = ["Rainfall_mm", "Avg_Temperature_C", "Production_Tonnes", "MSP", "Export_Demand_Tonnes", "Prev_Year_Price"]
        categorical_features = ["Crop", "Region"]

        # One-hot encode categorical features
        input_encoded = encoder.transform(input_df[categorical_features]).toarray()  # Important for consistency
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_feature_names)

        logging.info(f"One-hot encoded categorical features:\n{input_encoded_df}")

        # Scale numerical columns
        input_scaled_num = scaler.transform(input_df[numerical_features])
        input_scaled_num_df = pd.DataFrame(input_scaled_num, columns=numerical_features)

        logging.info(f"Scaled numerical features:\n{input_scaled_num_df}")

        # Combine scaled numerics and encoded categoricals
        final_input_df = pd.concat([input_scaled_num_df, input_encoded_df], axis=1)

        logging.info(f"Final input for prediction (combined):\n{final_input_df}")

        # Ensure it's a numpy array before prediction
        final_input = final_input_df.values

        # Prediction
        prediction = model.predict(final_input)

        logging.info(f"Prediction result: {prediction[0]}")

        # Return the prediction (already a continuous value)
        return prediction[0]

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logging.error(error_msg)
        return error_msg
