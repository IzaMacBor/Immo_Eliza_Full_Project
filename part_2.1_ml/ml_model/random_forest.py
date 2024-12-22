import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class PricePredictionModel:
    def __init__(self, input_prefix, model_output_path):
        self.input_prefix = input_prefix
        self.model_output_path = model_output_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self):
        # Load preprocessed training and test data
        self.X_train = pd.read_csv(os.path.join(self.input_prefix, "X_train.csv"))
        self.X_test = pd.read_csv(os.path.join(self.input_prefix, "X_test.csv"))
        self.y_train = pd.read_csv(os.path.join(self.input_prefix, "y_train.csv")).values.ravel()
        self.y_test = pd.read_csv(os.path.join(self.input_prefix, "y_test.csv")).values.ravel()

    def train_model(self):
        # Initialize and train the Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

    def evaluate_model(self):
        # Predict on the test set
        y_pred = self.model.predict(self.X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"Model Evaluation Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R-squared (R2): {r2:.2f}")

    def save_model(self):
        # Save the trained model to a file
        joblib.dump(self.model, self.model_output_path)
        print(f"Model saved to {self.model_output_path}")

    def run_pipeline(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()
        self.save_model()

# Main script
if __name__ == "__main__":
    # Paths to input and model output
    input_dir = r"C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_2.1_ml\data\preprocessed_csv"
    model_output_file = r"C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_2.1_ml\model_output\random_forest_model.joblib"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_output_file), exist_ok=True)

    # Run the prediction pipeline
    model_pipeline = PricePredictionModel(input_dir, model_output_file)
    model_pipeline.run_pipeline()
