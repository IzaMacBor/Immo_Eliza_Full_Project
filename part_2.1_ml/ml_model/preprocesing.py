import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import re

class DataPreprocessor:
    def __init__(self, file_path, output_prefix):
        self.file_path = file_path
        self.output_prefix = output_prefix
        self.df = None

    def load_data(self):
        # Load data
        self.df = pd.read_csv(self.file_path)

    def clean_data(self):
        # Remove unnecessary columns
        self.df = self.df.drop(columns=["link", "property_id", "subtype_of_property", "type_of_sale", "locality_name", "latitude", "longitude"], errors="ignore")

        # Remove rows where postal_code is not in the range 0000-9999
        self.df = self.df[self.df['postal_code'].apply(lambda x: bool(re.match(r'^\d{4}$', str(x))))]

        # Fill missing values for numeric columns
        numeric_cols = ['terrace_surface', 'garden', 'facades', 'land_area', 'construction_year', 'nr_bathrooms']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)

        # Fill missing values for bedrooms with median
        if 'bedrooms' in self.df.columns:
            self.df['bedrooms'] = self.df['bedrooms'].fillna(self.df['bedrooms'].median())

        # Fill missing values in categorical columns with the most frequent value
        categorical_cols = ['furnished', 'equipped_kitchen', 'state_of_building']
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        # Map state_of_building to numeric values
        state_mapping = {
            'AS_NEW': 0,
            'JUST_RENOVATED': 0,
            'GOOD': 1,
            'TO_RENOVATE': 2,
            'TO_RESTORE': 3,
            'TO_BE_DONE_UP': 4
        }
        if 'state_of_building' in self.df.columns:
            self.df['state_of_building'] = self.df['state_of_building'].map(state_mapping)

        # Map equipped_kitchen to numeric values
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
        if 'equipped_kitchen' in self.df.columns:
            self.df['equipped_kitchen'] = self.df['equipped_kitchen'].map(equipped_kitchen_mapping)

        # Map type_of_property to numeric values
        property_type_mapping = {
            'HOUSE': 1,
            'APARTMENT': 0
        }
        if 'type_of_property' in self.df.columns:
            self.df['type_of_property'] = self.df['type_of_property'].map(property_type_mapping)

        # Convert `furnished` to binary (1 if furnished, 0 otherwise)
        if 'furnished' in self.df.columns:
            self.df['furnished'] = self.df['furnished'].apply(lambda x: 1 if x == 1.0 else 0)

    def encode_and_scale(self):
        # Scale numerical data
        numeric_cols = ['living_area', 'terrace_surface', 'garden', 'facades', 'land_area', 'price', 
                        'construction_year', 'bedrooms', 'nr_bathrooms']
        numeric_cols = [col for col in numeric_cols if col in self.df.columns]  # Check if columns exist
        scaler = StandardScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])

    def split_and_save(self):
        # Split into input variables (X) and target (y)
        y = self.df['price']
        X = self.df.drop(columns=['price'])

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Save the prepared data
        X_train.to_csv(os.path.join(self.output_prefix, f"X_train.csv"), index=False)
        X_test.to_csv(os.path.join(self.output_prefix, f"X_test.csv"), index=False)
        y_train.to_csv(os.path.join(self.output_prefix, f"y_train.csv"), index=False)
        y_test.to_csv(os.path.join(self.output_prefix, f"y_test.csv"), index=False)

        print(f"Data processed and saved as: X_train, X_test, y_train, y_test")

    def preprocess(self):
        self.load_data()
        self.clean_data()
        self.encode_and_scale()
        self.split_and_save()

# Main part of the script
if __name__ == "__main__":
    # Hardcoded paths for the raw properties file and output directory
    raw_properties_file = r"C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_1_scraping\data\raw_properties.csv"
    output_dir = r"C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_2.1_ml\data\preprocessed_csv"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Data processing
    processor = DataPreprocessor(raw_properties_file, output_dir)
    processor.preprocess()