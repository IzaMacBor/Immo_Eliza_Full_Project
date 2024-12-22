import streamlit as st
import pandas as pd
from joblib import load


class RealEstateModel:
    def __init__(self, model_path: str):
        self.model = load(model_path)

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)


class InputMapper:
    def __init__(self):
        self.property_type_mapping = {'HOUSE': 1, 'APARTMENT': 0}
        self.equipped_kitchen_mapping = {
            'INSTALLED': 0,
            'HYPER_EQUIPPED': 1,
            'NOT_INSTALLED': 2,
            'USA_INSTALLED': 3,
            'SEMI_EQUIPPED': 4,
            'USA_HYPER_EQUIPPED': 5,
            'USA_SEMI_EQUIPPED': 6,
            'USA_UNINSTALLED': 7
        }
        self.state_mapping = {
            'AS_NEW': 0,
            'JUST_RENOVATED': 0,
            'GOOD': 1,
            'TO_RENOVATE': 2,
            'TO_RESTORE': 3,
            'TO_BE_DONE_UP': 4
        }

    def map_inputs(self, inputs: dict) -> pd.DataFrame:
        data = {
            'postal_code': [inputs['postal_code']],
            'type_of_property': [self.property_type_mapping[inputs['type_of_property']]],
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
            'equipped_kitchen': [self.equipped_kitchen_mapping[inputs['equipped_kitchen']]],
            'state_of_building': [self.state_mapping[inputs['state_of_building']]]
        }
        return pd.DataFrame(data)


class RealEstateApp:
    def __init__(self, model_path: str):
        self.model = RealEstateModel(model_path)
        self.mapper = InputMapper()

    def run(self):
        st.title("Real Estate Price Prediction")
        st.header("Enter property details:")

        # Input fields
        inputs = self.get_user_inputs()

        # Predict button
        if st.button("Predict Price"):
            processed_data = self.mapper.map_inputs(inputs)
            prediction = self.model.predict(processed_data)
            st.success(f"Predicted Price: €{prediction[0]:,.2f}")

    @staticmethod
    def get_user_inputs() -> dict:
        postal_code = st.number_input("Postal Code", min_value=1000, max_value=9999, step=1)
        type_of_property = st.selectbox("Type of Property", ["HOUSE", "APARTMENT"])
        construction_year = st.number_input("Construction year", min_value=1700, max_value=2100, step=1)
        living_area = st.number_input("Living Area (m²)", min_value=0, step=1)
        bedrooms = st.number_input('Number of bedrooms', min_value=1, step=1)
        nr_bathrooms = st.number_input('Number of bathrooms', min_value=1, step=1)
        furnished = st.selectbox("Furnished", ["Yes", "No"])
        open_fire = st.selectbox("Open Fire", ["Yes", "No"])
        terrace_surface = st.number_input("Terrace Surface (m²)", min_value=0, step=1)
        garden = st.selectbox("Garden", ["Yes", "No"])
        facades = st.number_input("Number of Facades", min_value=1, max_value=4, step=1)
        swimming_pool = st.selectbox("Swimming Pool", ["Yes", "No"])
        land_area = st.number_input("Land Area (m²)", min_value=0, step=1)
        equipped_kitchen = st.selectbox("Equipped Kitchen Type", [
            'INSTALLED', 'HYPER_EQUIPPED', 'NOT_INSTALLED', 'USA_INSTALLED', 
            'SEMI_EQUIPPED', 'USA_HYPER_EQUIPPED', 'USA_SEMI_EQUIPPED', 'USA_UNINSTALLED'
        ])
        state_of_building = st.selectbox("State of Building", [
            'AS_NEW', 'JUST_RENOVATED', 'GOOD', 'TO_RENOVATE', 'TO_RESTORE', 'TO_BE_DONE_UP'
        ])

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


if __name__ == "__main__":
    app = RealEstateApp(r'C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_2.1_ml\model_output\random_forest_model.joblib')
    app.run()
