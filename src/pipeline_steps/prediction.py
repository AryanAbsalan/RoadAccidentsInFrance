import joblib
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path("models/model.joblib"))

    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction