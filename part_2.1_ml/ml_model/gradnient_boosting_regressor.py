from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os
import joblib

class RealEstateModel:
    def __init__(self, X_train_path, X_test_path, y_train_path, y_test_path, output_dir):
        self.X_train = pd.read_csv(X_train_path)
        self.X_test = pd.read_csv(X_test_path)
        self.y_train = pd.read_csv(y_train_path).squeeze()
        self.y_test = pd.read_csv(y_test_path).squeeze()
        self.output_dir = output_dir
        self.model = GradientBoostingRegressor(random_state=42)

    def handle_missing_values(self):
        # Using SimpleImputer to fill missing values in X_train and X_test
        imputer = SimpleImputer(strategy='mean')  # or 'median' depending on your choice
        self.X_train = imputer.fit_transform(self.X_train)
        self.X_test = imputer.transform(self.X_test)

        # Handle missing values in target variables (y)
        if self.y_train.isnull().any():
            print("Missing values in y_train. Replacing with the mean...")
            self.y_train.fillna(self.y_train.mean(), inplace=True)
        
        if self.y_test.isnull().any():
            print("Missing values in y_test. Replacing with the mean...")
            self.y_test.fillna(self.y_test.mean(), inplace=True)

    def tune_hyperparameters(self):
        # Define the parameter grid for GradientBoostingRegressor
        param_grid = {
            'regressor__n_estimators': [100, 150, 200],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 4, 5],
            'regressor__min_samples_split': [2, 5, 10]
        }

        # Create a pipeline with SimpleImputer and GradientBoostingRegressor
        pipeline = Pipeline(steps=[ 
            ('imputer', SimpleImputer(strategy='mean')), 
            ('regressor', GradientBoostingRegressor(random_state=42))
        ])
        
        # Use GridSearchCV to find the best parameters
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(self.X_train, self.y_train)
        
        # Best parameters found
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Set the model to the best one
        self.model = grid_search.best_estimator_

    def train(self):
        print("Training the model...")
        self.model.fit(self.X_train, self.y_train)

        # Save the model
        model_path = os.path.join(self.output_dir, "gradient_boosting_model_optimized.joblib")
        joblib.dump(self.model, model_path)
        print(f"Model saved at {model_path}")

    def evaluate(self):
        # Evaluate the model on the training data
        train_predictions = self.model.predict(self.X_train)
        train_mse = mean_squared_error(self.y_train, train_predictions)
        train_mae = mean_absolute_error(self.y_train, train_predictions)
        train_r2 = r2_score(self.y_train, train_predictions)

        # Evaluate the model on the testing data
        test_predictions = self.model.predict(self.X_test)
        test_mse = mean_squared_error(self.y_test, test_predictions)
        test_mae = mean_absolute_error(self.y_test, test_predictions)
        test_r2 = r2_score(self.y_test, test_predictions)

        # Print metrics
        print("Metrics for the Gradient Boosting (optimized) model:")
        print(f"Training MSE: {train_mse}")
        print(f"Training MAE: {train_mae}")
        print(f"Training R2: {train_r2}")
        print(f"Test MSE: {test_mse}")
        print(f"Test MAE: {test_mae}")
        print(f"Test R2: {test_r2}")

        # Save the evaluation results to a file
        evaluation_results = {
            "Train MSE": train_mse,
            "Train MAE": train_mae,
            "Train R2": train_r2,
            "Test MSE": test_mse,
            "Test MAE": test_mae,
            "Test R2": test_r2
        }
        results_path = os.path.join(self.output_dir, "model_evaluation_results_optimized.csv")
        pd.DataFrame([evaluation_results]).to_csv(results_path, index=False)
        print(f"Evaluation results saved at {results_path}")

    def predict(self, X_input):
        return self.model.predict(X_input)

# Main part of the script
if __name__ == "__main__":
    X_train_path = r"C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_2.1_ml\data\X_train.csv"
    X_test_path = r"C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_2.1_ml\data\X_test.csv"
    y_train_path = r"C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_2.1_ml\data\y_train.csv"
    y_test_path = r"C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_2.1_ml\data\y_test.csv"
    output_dir = r"C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_2.1_ml\data\model_output"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize and run the model
    model = RealEstateModel(X_train_path, X_test_path, y_train_path, y_test_path, output_dir)
    model.handle_missing_values()
    model.tune_hyperparameters()  # Hyperparameter tuning
    model.train()
    model.evaluate()
