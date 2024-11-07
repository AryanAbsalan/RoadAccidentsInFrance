import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import joblib
from pathlib import Path
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
import os
# Insert the absolute path of the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.entity import ModelEvaluationConfig
from src.common_utils import save_json
from custom_logger import logger
import json

# To fill in with your repo information
dagshub.init(repo_owner='aryan.absalan', repo_name='RoadAccidentsInFrance', mlflow=True)

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted')
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
        return accuracy, precision, recall, f1
    
    def log_into_mlflow(self):
        X_test = pd.read_csv(self.config.X_test_path)
        y_test = pd.read_csv(self.config.y_test_path)
        model = joblib.load(self.config.model_path)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predictions = model.predict(X_test)

            # Calculate classification metrics
            accuracy, precision, recall, f1 = self.eval_metrics(y_test, predictions)
            
            # Save metrics locally
            scores = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Generate and save the evaluation report
            self.generate_report(accuracy, precision, recall, f1)

            # Log parameters and metrics to MLflow
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            logger.info("accuracy: %s,  precision: %s, recall: %s, f1_score: %s", accuracy, precision, recall, f1)

            # Log the model
            try:
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="DecisionTreeModel", input_example=X_test.iloc[0].to_dict())
                    print("Model logged successfully.")
                else:
                    mlflow.sklearn.log_model(model, "model")
            except Exception as e:
                print(f"Error while logging model: {str(e)}")
                logger.exception(f"Error while logging model: {str(e)}")

    def generate_report(self, accuracy, precision, recall, f1):
        report_dir = Path("reports")
        report_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        report_path = report_dir / "decision_tree_evaluation_report.txt"
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n")
        logger.info(f"Report saved at {report_path}")
    def generate_report(self, accuracy, precision, recall, f1):
        # Directory for reports
        report_dir = Path("reports")
        report_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        # Path for the JSON report
        json_report_path = report_dir / "decision_tree_evaluation_report.json"
        report_data = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        # Writing to JSON file
        with open(json_report_path, "w", encoding='utf-8') as json_file:
            json.dump(report_data, json_file, indent=4)
        logger.info(f"JSON report saved at {json_report_path}")
        
        # Path for the text report
        report_path = report_dir / "decision_tree_evaluation_report.txt"
        with open(report_path, "w", encoding='utf-8') as f:
                        f.write("Accuracy: {} ---- Precision: {} ---- Recall: {} ---- F1 Score: {}".format(accuracy, precision, recall, f1))
        logger.info("Text report saved at {}".format(report_path))
            

