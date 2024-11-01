import os
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from src.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        X_train = pd.read_csv(self.config.X_train_path)
        y_train = pd.read_csv(self.config.y_train_path)

        dtc = DecisionTreeClassifier(class_weight=self.config.class_weight, 
                                    max_depth=self.config.max_depth, 
                                    min_samples_leaf=self.config.min_samples_leaf, 
                                    min_samples_split=self.config.min_samples_split)
        dtc.fit(X_train, y_train)
        # Ensure output directory exists
        os.makedirs(self.config.root_dir, exist_ok=True)  # Create directory if it doesn't exist
        joblib.dump(dtc, os.path.join(self.config.root_dir, self.config.model_name))