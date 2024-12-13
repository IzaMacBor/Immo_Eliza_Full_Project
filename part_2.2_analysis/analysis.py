import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class RealEstateMarket:
    def __init__(self, data_path, output_folder):
        """Initialize the RealEstateMarket class."""
        self.data_path = data_path
        self.output_folder = output_folder
        self.data = None

        # Ensure the output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

    def load_data(self):
        """Load the real estate data from a CSV file."""
        self.data = pd.read_csv(self.data_path)
        print("Data loaded successfully.")

    def preprocess_data(self):
        """Perform basic preprocessing on the data."""
        # Fill missing values for numeric columns with 0
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(0)

        # Fill missing values for categorical columns with 'UNKNOWN'
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        self.data[categorical_cols] = self.data[categorical_cols].fillna('UNKNOWN')

        # Remove rows where living_area or price is zero
        self.data = self.data[(self.data['living_area'] > 0) & (self.data['price'] > 0)]

        print("Data preprocessing completed.")

    def handle_outliers(self, column, threshold=1.5):
        """Handle outliers in a given column using the IQR method."""
        if column not in self.data.columns:
            print(f"Column {column} does not exist in the data.")
            return

        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        print(f"Detected {len(outliers)} outliers in {column}.")

        # Optionally remove outliers
        self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
        print(f"Outliers in {column} removed.")

    def analyze_prices_by_type(self):
        """Analyze and visualize average prices by type of property."""
        avg_price_by_type = self.data.groupby('type_of_property')['price'].mean().sort_values()

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x=avg_price_by_type.index, y=avg_price_by_type.values)
        plt.title('Average Property Prices by Type')
        plt.xlabel('Type of Property')
        plt.ylabel('Average Price (€)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_file = os.path.join(self.output_folder, 'average_prices_by_type.jpg')
        plt.savefig(output_file)
        plt.close()
        print(f"Visualization saved to {output_file}")

    def analyze_area_vs_price(self):
        """Analyze the relationship between living area and price."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='living_area', y='price', hue='type_of_property', data=self.data, alpha=0.7)
        plt.title('Living Area vs Price')
        plt.xlabel('Living Area (m²)')
        plt.ylabel('Price (€)')
        plt.legend(title='Type of Property')
        plt.tight_layout()
        output_file = os.path.join(self.output_folder, 'living_area_vs_price.jpg')
        plt.savefig(output_file)
        plt.close()
        print(f"Visualization saved to {output_file}")

    def analyze_prices_by_location_combined(self, top_n=30):
        """Analyze average prices per square meter by location, showing both most and least expensive locations."""
        if 'living_area' in self.data.columns and 'price' in self.data.columns:
            self.data['price_per_sqm'] = self.data['price'] / self.data['living_area']
        else:
            print("Required columns ('living_area', 'price') not found.")
            return

        avg_price_location = self.data.groupby('locality_name')['price_per_sqm'].mean().sort_values()
        top_locations = avg_price_location.tail(top_n)
        bottom_locations = avg_price_location.head(top_n)

        # Most and least expensive locations
        print("Most expensive locations:")
        print(top_locations)
        print("\nLeast expensive locations:")
        print(bottom_locations)

        # Combined plot for both top and bottom locations
        plt.figure(figsize=(12, 8))

        # Plotting most expensive locations
        plt.subplot(1, 2, 1)
        sns.barplot(x=top_locations.values, y=top_locations.index)
        plt.title(f'Top {top_n} Most Expensive Locations by Price per Square Meter')
        plt.xlabel('Price per m² (€)')
        plt.ylabel('Location')

        # Plotting least expensive locations
        plt.subplot(1, 2, 2)
        sns.barplot(x=bottom_locations.values, y=bottom_locations.index)
        plt.title(f'Top {top_n} Least Expensive Locations by Price per Square Meter')
        plt.xlabel('Price per m² (€)')
        plt.ylabel('Location')

        plt.tight_layout()
        output_file = os.path.join(self.output_folder, 'prices_by_location_combined.jpg')
        plt.savefig(output_file)
        plt.close()
        print(f"Visualization saved to {output_file}")

    def analyze_correlations(self):
        """Analyze correlations between numeric variables."""
        # Select only numeric columns for correlation analysis
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])

        if numeric_data.empty:
            print("No numeric data available for correlation analysis.")
            return

        excluded_columns = self.data.select_dtypes(exclude=['float64', 'int64']).columns
        print(f"Excluding non-numeric columns: {excluded_columns.tolist()}")

        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()

        # Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title('Correlation Matrix')
        plt.tight_layout()
        output_file = os.path.join(self.output_folder, 'correlation_matrix.jpg')
        plt.savefig(output_file)
        plt.close()
        print(f"Visualization saved to {output_file}")

    def save_cleaned_data(self, output_path):
        """Save the cleaned and preprocessed data to a CSV file."""
        self.data.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}.")

# Example usage:
data_path = r'C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_1_scraping\data\raw_properties.csv'
output_folder = r'C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_2.2_analysis\vizualizations'
output_path = r'C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_2.2_analysis\data\cleaned_data.csv'

# Create an instance of the RealEstateMarket class
market_analysis = RealEstateMarket(data_path, output_folder)

# Load the data
market_analysis.load_data()

# Preprocess the data
market_analysis.preprocess_data()

# Handle outliers in price and living_area columns
market_analysis.handle_outliers('price')
market_analysis.handle_outliers('living_area')

# Analyze average prices by property type
market_analysis.analyze_prices_by_type()

# Analyze the relationship between living area and price
market_analysis.analyze_area_vs_price()

# Analyze average prices by location (combined most and least expensive locations)
market_analysis.analyze_prices_by_location_combined(top_n=30)

# Analyze correlations
market_analysis.analyze_correlations()

# Save the cleaned data
market_analysis.save_cleaned_data(output_path)
