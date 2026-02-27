from pathlib import Path
import src.classifier as c

model_path = Path("models/random_forest_v1.joblib")
WallClassifier = c.WallClassifier()