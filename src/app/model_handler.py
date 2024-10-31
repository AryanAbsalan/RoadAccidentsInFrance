import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# Load environment variables from .env file
load_dotenv()

class ModelHandler:
    def __init__(self):
        """Initialize ModelHandler with paths and load the dataset."""
        self.model_path = os.getenv("MODEL_PATH")
        self.preprocessor_path = os.getenv("PREPROCESSOR_PATH")
        self.dataset_path = os.getenv("DATASET_PATH")
        self.dataset_name = os.getenv("DATASET_NAME")
        self.model = None
        self.preprocessor = None
        self.df = None  

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset from the specified path."""
        full_dataset_path = os.path.join(self.dataset_path, self.dataset_name)
        try:
            self.df = pd.read_csv(full_dataset_path)
            print("Dataset loaded successfully.")
            return self.df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def split_data(self):
        """Split the dataset into training and testing sets."""
        if self.df is None:
            raise ValueError("Dataset not loaded. Please load the dataset first.")

        X = self.df.drop(columns=['Severity'])
        y = self.df['Severity']
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def create_preprocessor(self):
        """Create the column transformer for preprocessing."""
        categorical_features = [
            'Safety_Equipment', 'Department_Code', 'Mobile_Obstacle',
            'Vehicle_Category', 'Position_In_Vehicle', 'Collision_Type', 
            'Time_of_Day', 'Journey_Type', 'Obstacle_Hit', 
            'Road_Category', 'Gender', 'User_Category', 'Intersection_Type'
        ]
        numerical_features = ['Driver_Age', 'Number_of_Lanes']

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ]
        )

    def preprocess_data(self, X_train):
        """Preprocess the training data."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not created. Please create the preprocessor first.")
        
        return self.preprocessor.fit_transform(X_train)

    def save_preprocessor(self):
        """Save the preprocessor to a file."""
        if self.preprocessor:
            joblib.dump(self.preprocessor, self.preprocessor_path)
            print(f"Preprocessor saved successfully at {self.preprocessor_path}")
        else:
            raise ValueError("No preprocessor to save.")

    def train_model(self, X_train, y_train):
        """Train the decision tree model."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not created. Please create the preprocessor first.")

        decision_tree_model = DecisionTreeClassifier(class_weight='balanced', max_depth=10, min_samples_leaf=5, min_samples_split=5)
        decision_tree_model.fit(X_train, y_train)
        self.model = decision_tree_model

    def save_model(self):
        """Save the trained model to a file."""
        if self.model:
            joblib.dump(self.model, self.model_path)
            print(f"Model saved successfully at {self.model_path}")
        else:
            raise ValueError("No model to save.")
    
    def predict(self, features: list) -> Any:
        """Transform input features and make a prediction using the loaded model."""
        if self.model and self.preprocessor:
            try:
                # Define the expected feature columns
                feature_columns = [
                    'Driver_Age', 'Safety_Equipment', 'Department_Code', 'Mobile_Obstacle', 
                    'Vehicle_Category', 'Position_In_Vehicle', 'Collision_Type', 'Number_of_Lanes', 
                    'Time_of_Day', 'Journey_Type', 'Obstacle_Hit', 'Road_Category', 'Gender', 
                    'User_Category', 'Intersection_Type'
                ]
                
                # Create a DataFrame with correct column names
                features_df = pd.DataFrame([features], columns=feature_columns)  # Wrap in a list to make it a single row
                
                # Apply the preprocessor transformation
                transformed_features = self.preprocessor.transform(features_df)

                # Predict using the transformed features
                prediction = self.model.predict(transformed_features)
                
                return prediction[0]  # Return the first prediction
            except Exception as e:
                print(f"Error during prediction: {e}")
                raise
        else:
            raise ValueError("Model or preprocessor is not loaded.")
        
    
    @staticmethod
    def execute_pipeline():
        """Static method to execute the entire pipeline."""
        handler = ModelHandler()
        
        # Perform steps
        df = handler.load_dataset()
        X_train, X_test, y_train, y_test = handler.split_data()
        handler.create_preprocessor()
        X_train_preprocessed = handler.preprocess_data(X_train)
        handler.save_preprocessor()
        handler.train_model(X_train_preprocessed, y_train)
        handler.save_model()
        print("Model training and saving complete.")

# Main function to execute the steps
if __name__ == "__main__":
    ModelHandler.execute_pipeline()
    
