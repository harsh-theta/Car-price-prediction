import streamlit as st
import pickle
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer


# Load encoders and models
# with open('models/one_hot_encoder.pkl', 'rb') as file:
#     oh_encoder = pickle.load(file)

# with open('models/binary_encoder.pkl', 'rb') as file:
#     binary_encoder = pickle.load(file)

# with open('models/standard_scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)

with open('models/regression_model.pkl', 'rb') as file:
    regression_model = pickle.load(file)

with open('models/random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('models/preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# Load car name mapping
with open('dataset/car-name-mapping.json', 'r') as file:
    car_name_mapping = json.load(file)

# Streamlit app
st.title("Car Price Predictor")

# Input fields
car_name = st.selectbox("Select Car Name", options=list(car_name_mapping.keys()))
vehicle_age = st.number_input("Vehicle Age (in years)", min_value=0, max_value=50, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
mileage = st.number_input("Mileage (kmpl)", min_value=0.0, step=0.1, format="%.1f")
engine = st.number_input("Engine Capacity (CC)", min_value=0, step=10)
max_power = st.number_input("Max Power (BHP)", min_value=0.0, step=0.1, format="%.1f")
seats = st.number_input("Number of Seats", min_value=2, max_value=10, step=1)
seller_type = st.selectbox("Seller Type", options=["Individual", "Dealer"])
fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission_type = st.selectbox("Transmission Type", options=["Manual", "Automatic"])


num_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
onehot_columns = ['seller_type','fuel_type','transmission_type']
binary_columns = ['car_name']



# Validate inputs
if st.button("Predict Price"):
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            "car_name": [car_name_mapping[car_name]],
            "vehicle_age": [vehicle_age],
            "km_driven": [km_driven],
            "mileage": [mileage],
            "engine": [engine],
            "max_power": [max_power],
            "seats": [seats],
            "seller_type": [seller_type],
            "fuel_type": [fuel_type],
            "transmission_type": [transmission_type]
        })

        
        input_data = preprocessor.transform(input_data)
        # # Apply transformations
        # input_data = binary_encoder.transform(input_data)
        # input_data = oh_encoder.transform(input_data)
        # input_data = scaler.transform(input_data)

        # Predict using both models
        regression_price = regression_model.predict(input_data)[0]
        rf_price = rf_model.predict(input_data)[0]

        # Display results
        st.success(f"Predicted Price (Linear Regression): ₹{regression_price:,.2f}")
        st.success(f"Predicted Price (Random Forest): ₹{rf_price:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

