import streamlit as st
import pandas as pd
from joblib import load
import json

# Function to make predictions and reverse standardization
def predict_price(input_features, model, price_mean, price_std):
    # Ensure input features are passed as a 2D array
    input_features = pd.DataFrame(input_features)  # Convert to DataFrame if not already
    prediction = model.predict(input_features)  # Make the prediction
    # Reverse standardization: (predicted_value * price_std) + price_mean
    prediction_original = (prediction[0] * price_std) + price_mean
    return prediction_original

# Load the model and stats
def load_model_and_stats(model_path, stats_path):
    model = load(model_path)
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return model, stats['price_mean'], stats['price_std']

# Streamlit app layout and prediction logic
def run_app():
    st.title("Real Estate Price Prediction")
    st.header("Enter property details:")

    # Input fields
    input_data = get_user_inputs()

    # Load model and stats
    model_path = r'C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_2.1_ml\model_output\random_forest_model.joblib'  # Update with your actual path
    stats_path = r'C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_2.1_ml\data\preprocessed_csv\stats.json'  # Update with your actual path
    model, price_mean, price_std = load_model_and_stats(model_path, stats_path)

    if st.button("Predict Price"):
        try:
            # Display the input data
            st.write("Input data:", input_data)

            # Map inputs to the feature set expected by the model
            processed_data = map_inputs(input_data)

            # Make the prediction using the model
            prediction = predict_price(processed_data, model, price_mean, price_std)
            st.success(f"Predicted Property Price: €{prediction:,.2f}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Map input data to model features
def map_inputs(inputs):
    property_type_mapping = {'HOUSE': 1, 'APARTMENT': 0}
    equipped_kitchen_mapping = {
        'INSTALLED': 0,
        'HYPER_EQUIPPED': 1,
        'NOT_INSTALLED': 2,
        'USA_INSTALLED': 3,
        'SEMI_EQUIPPED': 4,
        'USA_HYPER_EQUIPPED': 5,
        'USA_SEMI_EQUIPPED': 6,
        'USA_UNINSTALLED': 7
    }
    state_mapping = {
        'AS_NEW': 0,
        'JUST_RENOVATED': 0,
        'GOOD': 1,
        'TO_RENOVATE': 2,
        'TO_RESTORE': 3,
        'TO_BE_DONE_UP': 4
    }

    data = {
        'postal_code': [inputs['postal_code']],
        'type_of_property': [property_type_mapping[inputs['type_of_property']]],
        'construction_year': [inputs['construction_year']],
        'living_area': [inputs['living_area']],
        'bedrooms': [inputs['bedrooms']],
        'nr_bathrooms': [inputs['nr_bathrooms']],
        'furnished': [1 if inputs['furnished'] == "Yes" else 0],
        'open_fire': [1 if inputs['open_fire'] == "Yes" else 0],
        'terrace_surface': [inputs['terrace_surface']],
        'garden': [1 if inputs['garden'] == "Yes" else 0],
        'facades': [inputs['facades']],
        'swimming_pool': [1 if inputs['swimming_pool'] == "Yes" else 0],
        'land_area': [inputs['land_area']],
        'equipped_kitchen': [equipped_kitchen_mapping[inputs['equipped_kitchen']]],
        'state_of_building': [state_mapping[inputs['state_of_building']]]
    }
    return pd.DataFrame(data)

# Get user inputs
def get_user_inputs():
    postal_code = st.number_input("Postal Code", min_value=1000, max_value=9999, step=1, value=2300)
    type_of_property = st.selectbox("Type of Property", ["HOUSE", "APARTMENT"], index=0)
    construction_year = st.number_input("Construction year", min_value=1700, max_value=2100, step=1, value=1967)
    living_area = st.number_input("Living Area (m²)", min_value=0, step=1, value=200)
    bedrooms = st.number_input('Number of bedrooms', min_value=1, step=1, value=3)
    nr_bathrooms = st.number_input('Number of bathrooms', min_value=1, step=1, value=2)
    furnished = st.selectbox("Furnished", ["Yes", "No"], index=1)
    open_fire = st.selectbox("Open Fire", ["Yes", "No"], index=0)
    terrace_surface = st.number_input("Terrace Surface (m²)", min_value=0, step=1, value=20)
    garden = st.selectbox("Garden", ["Yes", "No"], index=0)
    facades = st.number_input("Number of Facades", min_value=1, max_value=4, step=1, value=3)
    swimming_pool = st.selectbox("Swimming Pool", ["Yes", "No"])
    land_area = st.number_input("Land Area (m²)", min_value=0, step=1, value=260)
    equipped_kitchen = st.selectbox("Equipped Kitchen Type", [
        'INSTALLED', 'HYPER_EQUIPPED', 'NOT_INSTALLED', 'USA_INSTALLED', 
        'SEMI_EQUIPPED', 'USA_HYPER_EQUIPPED', 'USA_SEMI_EQUIPPED', 'USA_UNINSTALLED'
    ], index=0)
    state_of_building = st.selectbox("State of Building", [
        'AS_NEW', 'JUST_RENOVATED', 'GOOD', 'TO_RENOVATE', 'TO_RESTORE', 'TO_BE_DONE_UP'
    ], index=1)

    return {
        'postal_code': postal_code,
        'type_of_property': type_of_property,
        'construction_year': construction_year,
        'living_area': living_area,
        'bedrooms': bedrooms,
        'nr_bathrooms': nr_bathrooms,
        'furnished': furnished,
        'open_fire': open_fire,
        'terrace_surface': terrace_surface,
        'garden': garden,
        'facades': facades,
        'swimming_pool': swimming_pool,
        'land_area': land_area,
        'equipped_kitchen': equipped_kitchen,
        'state_of_building': state_of_building
    }

# Run the app
if __name__ == "__main__":
    run_app()
