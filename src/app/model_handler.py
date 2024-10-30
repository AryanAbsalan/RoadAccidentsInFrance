import os
import joblib
from dotenv import load_dotenv
from typing import Any

# Load environment variables from .env file
load_dotenv()

class ModelHandler:
    def __init__(self, model_path: str):
        """Initialize ModelHandler with the given model path and load the model."""
        self.model_path = model_path
        self.model = self.load_model()  # Load the model upon initialization

    def load_model(self) -> Any:
        """Load the trained model from the specified model path."""
        try:
            model = joblib.load(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {e}")
            raise

    def predict(self, features: list) -> Any:
        """Make a prediction using the loaded model."""
        if self.model:
            try:
                # Predicting based on input features
                prediction = self.model.predict([features])
                return prediction[0]  # Return the first prediction
            except Exception as e:
                print(f"Error during prediction: {e}")
                raise
        else:
            raise ValueError("Model is not loaded.")

# Initialize the handler separately if needed
def initialize_handler(model_path: str) -> ModelHandler:
    """Initialize the ModelHandler with the model path."""
    return ModelHandler(model_path)
