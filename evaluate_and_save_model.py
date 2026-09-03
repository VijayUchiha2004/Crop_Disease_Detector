import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "crop_disease_model.h5"))
VAL_PATH = Path(os.getenv("VAL_DIR", BASE_DIR / "dataset" / "val"))
print("Loading validation data from:", VAL_PATH.resolve())

# Load validation dataset
val_ds = image_dataset_from_directory(
    str(VAL_PATH),
    image_size=(224, 224),
    batch_size=32,
    shuffle=False  # Do not shuffle for evaluation
)

# Get class names
class_names = val_ds.class_names
print("Detected classes:", class_names)

# Load the trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Set MODEL_PATH or train the model first.")

model = load_model(str(MODEL_PATH))
print("Model loaded successfully.")

# Evaluate the model
loss, accuracy = model.evaluate(val_ds)
print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Predict on validation data
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Get true labels
y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)

# Handle case where some classes are missing in predictions
labels = unique_labels(y_true, y_pred)
label_names = [class_names[i] for i in labels]

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=labels, target_names=label_names))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=labels))

# Save evaluation report
with open("evaluation_report.txt", "w") as f:
    f.write(f"Validation Loss: {loss:.4f}\n")
    f.write(f"Validation Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_true, y_pred, labels=labels, target_names=label_names))
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(confusion_matrix(y_true, y_pred, labels=labels)))
