import joblib
import pandas as pd
from dotenv import load_dotenv
from typing import Any

load_dotenv()

class ModelHandler:
    def __init__(self, model_path: str, preprocessor_path: str):
        """Initialize ModelHandler with model and preprocessor paths and load them."""
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = self.load_model()          # Load the model upon initialization
        self.preprocessor = self.load_preprocessor()  # Load the preprocessor

    def load_model(self) -> Any:
        """Load the trained model from the specified model path."""
        try:
            model = joblib.load(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {e}")
            raise

    def load_preprocessor(self) -> Any:
        """Load the preprocessor from the specified path."""
        try:
            preprocessor = joblib.load(self.preprocessor_path)
            print(f"Preprocessor loaded successfully from {self.preprocessor_path}")
            return preprocessor
        except Exception as e:
            print(f"Error loading preprocessor from {self.preprocessor_path}: {e}")
            raise

    def predict(self, features: list) -> Any:
        """Transform input features and make a prediction using the loaded model."""
        if self.model and self.preprocessor:
            try:
                """Make a prediction using the loaded model and preprocessor."""

                # Convert input features to a DataFrame with explicit column names
                feature_columns = [
                    'Driver_Age', 'Safety_Equipment', 'Department_Code', 'Mobile_Obstacle', 
                    'Vehicle_Category', 'Position_In_Vehicle', 'Collision_Type', 'Number_of_Lanes', 
                    'Time_of_Day', 'Journey_Type', 'Obstacle_Hit', 'Road_Category', 'Gender', 
                    'User_Category', 'Intersection_Type'
                ]
                
                # Create a DataFrame with correct column names
                features_df = pd.DataFrame([features], columns=feature_columns)  # Wrap in a list to make it a single row
                
                # Apply the preprocessor transformation
                try:
                    transformed_features = self.preprocessor.transform(features_df)
                except Exception as e:
                    print(f"Error during transformation: {e}")
                    raise

                # Predict using the transformed features
                prediction = self.model.predict(transformed_features)
                
                return prediction[0]  # Return the first prediction
            except Exception as e:
                print(f"Error during prediction: {e}")
                raise
        else:
            raise ValueError("Model or preprocessor is not loaded.")

