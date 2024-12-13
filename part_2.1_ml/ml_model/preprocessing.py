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
        self.df = self.df.drop(columns=["link", "property_id", "subtype_of_property", "type_of_sale", "locality_name"], errors="ignore")

        # Remove rows where postal_code is not in the range 0000-9999
        self.df = self.df[self.df['postal_code'].apply(lambda x: bool(re.match(r'^\d{4}$', str(x))))]

        # Fill missing values
        self.df['terrace_surface'] = self.df['terrace_surface'].fillna(0)
        self.df['garden'] = self.df['garden'].fillna(0)
        self.df['facades'] = self.df['facades'].fillna(0)
        self.df['land_area'] = self.df['land_area'].fillna(0)

        # Fill missing values in categorical columns with the most frequent value
        categorical_cols = ['furnished', 'equipped_kitchen', 'state_of_building']
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        # Mapping state_of_building to numeric values
        state_mapping = {
            'AS_NEW': 0,
            'JUST_RENOVATED': 0,
            'GOOD': 1,
            'TO_RENOVATE': 2,
            'TO_RESTORE': 3,
            'TO_BE_DONE_UP': 4
        }

        # Apply the mapping to the 'state_of_building' column
        if 'state_of_building' in self.df.columns:
            self.df['state_of_building'] = self.df['state_of_building'].map(state_mapping)

        # Mapping equipped_kitchen to numeric values
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

        # Apply the mapping to the 'equipped_kitchen' column
        if 'equipped_kitchen' in self.df.columns:
            self.df['equipped_kitchen'] = self.df['equipped_kitchen'].map(equipped_kitchen_mapping)

        # Mapping type_of_property to numeric values (House: 1, Apartment: 0)
        property_type_mapping = {
            'HOUSE': 1,
            'APARTMENT': 0
        }

        # Apply the mapping to the 'type_of_property' column
        if 'type_of_property' in self.df.columns:
            self.df['type_of_property'] = self.df['type_of_property'].map(property_type_mapping)

    def encode_and_scale(self):
        # One-Hot Encoding for categorical variables
        categorical_cols = ['type_of_property']  # No need to one-hot encode since it's already numeric
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)

        # Scale numerical data
        numeric_cols = ['living_area', 'terrace_surface', 'garden', 'facades', 'land_area', 'price']
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
    # Hardcoded paths for input and output files
    apartments_file = r"C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_1_scraping\data\scraped_csv\raw_apartments.csv"  # Replace with your actual file name
    houses_file = r"C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_1_scraping\data\scraped_csv\raw_houses.csv"        # Replace with your actual file name
    output_dir = r"C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_2.1_ml\data\preprocessed_csv"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Data processing
    apartments_processor = DataPreprocessor(apartments_file, output_dir)
    apartments_processor.preprocess()

    houses_processor = DataPreprocessor(houses_file, output_dir)
    houses_processor.preprocess()
