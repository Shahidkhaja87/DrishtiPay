
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "currency" / "test"
MODEL_PATH = ROOT / "models" / "best_currency_model.keras"
LABELS_PATH = ROOT / "models" / "class_names.json"
REPORT_PATH = ROOT / "models" / "evaluation_report.txt"
MATRIX_PATH = ROOT / "models" / "confusion_matrix.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

print("Loading saved model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    saved_labels = json.load(f)

print("Loading labelled test images...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = test_ds.class_names

if class_names != saved_labels:
    raise ValueError(
        f"Label mismatch!\nDataset: {class_names}\nSaved: {saved_labels}"
    )

def prepare(images, labels):
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels

test_ds = test_ds.map(prepare).prefetch(tf.data.AUTOTUNE)

print("Predicting test images...")
true_labels = np.concatenate(
    [labels.numpy() for _, labels in test_ds]
)
scores = model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(scores, axis=1)

accuracy = accuracy_score(true_labels, predicted_labels)
report = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
    digits=4,
)
matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
)

summary = (
    f"Test images: {len(true_labels)}\n"
    f"Test accuracy: {accuracy:.4%}\n\n"
    f"Classification report:\n{report}\n"
    f"Confusion matrix (rows=true, columns=predicted):\n"
    f"Labels order: {class_names}\n{matrix}\n"
)

print("\n" + summary)
REPORT_PATH.write_text(summary, encoding="utf-8")

np.savetxt(
    MATRIX_PATH,
    matrix,
    delimiter=",",
    fmt="%d",
    header="Rows=true labels; columns=predicted labels. Order: "
    + ", ".join(class_names),
    comments="",
)

print(f"\nReport saved to: {REPORT_PATH}")
print(f"Confusion matrix saved to: {MATRIX_PATH}")
