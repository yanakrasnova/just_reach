from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import src.utils as u


dataset_root = Path("data/dataset")
model_dir = Path("models")

X_train, y_train, classes = u.load_dataset(dataset_root / "train", expected_shape=(64, 64, 13))
X_test, y_test, classes = u.load_dataset(dataset_root / "test", expected_shape=(64, 64, 13))

X_train_preprocessed = u.preprocess_patches_dataset(X_train)
X_test_preprocessed = u.preprocess_patches_dataset(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train_preprocessed, y_train)

y_pred = rf.predict(X_test_preprocessed)
print("Evaluation on Test Set:")
print(classification_report(y_test, y_pred, target_names=classes))


model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "random_forest_v1.joblib"
joblib.dump(rf, model_path)
print(f"Model saved to {model_path}")