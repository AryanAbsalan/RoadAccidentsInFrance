import os
import pandas as pd
from sklearn.model_selection import train_test_split
from custom_logger import logger

from sklearn.model_selection import train_test_split
from src.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        X = data.drop(columns=["Severity"])
        y = data["Severity"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Ensure the output directory exists
        output_dir = os.path.dirname(self.config.transform_data_path)  # Get the directory from the full path
        os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Save the train-test split data as a DataFrame
        transformed_data = pd.concat([X_train, y_train], axis=1)
        transformed_data.to_csv(self.config.transform_data_path, index=False)  # Save to the full path

        X_train.to_csv(os.path.join(self.config.root_dir, "X_train.csv"), index = False)
        y_train.to_csv(os.path.join(self.config.root_dir, "y_train.csv"), index = False)
        X_test.to_csv(os.path.join(self.config.root_dir, "X_test.csv"), index = False)
        y_test.to_csv(os.path.join(self.config.root_dir, "y_test.csv"), index = False)

        logger.info("Splitted data into training and test sets")
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        logger.info(f"Transformed data saved at: {output_dir}")

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)