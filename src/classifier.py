import joblib
import os

class WallClassifier:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, features):
        # expects features of same shape as the model was trained on
        return self.model.predict(features)