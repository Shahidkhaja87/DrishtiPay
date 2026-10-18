
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
)
from tensorflow.keras import layers, models

# Reproducibility
SEED = 42
tf.keras.utils.set_random_seed(SEED)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "currency"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12

for split in ("train", "val", "test"):
    if not (DATA_DIR / split).exists():
        raise FileNotFoundError(
            f"Missing {DATA_DIR / split}. "
            "Create train, val and test folders first."
        )

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = train_ds.class_names
print("Class order:", class_names)

# Ensure every split has the same classes in the same order.
for split_name, ds in [("val", val_ds), ("test", test_ds)]:
    if ds.class_names != class_names:
        raise ValueError(
            f"{split_name} classes {ds.class_names} do not match "
            f"training classes {class_names}"
        )

if len(class_names) < 2:
    raise ValueError("Need at least two denomination classes.")

# MobileNetV2 expects input pixels scaled to [-1, 1].
def prepare(images, labels):
    return preprocess_input(images), labels

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(prepare).prefetch(AUTOTUNE)
val_ds = val_ds.map(prepare).prefetch(AUTOTUNE)
test_ds = test_ds.map(prepare).prefetch(AUTOTUNE)

# Data augmentation is applied only during training.
augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.04),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.10),
], name="augmentation")

base = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = augmentation(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(
    len(class_names), activation="softmax"
)(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

best_path = MODEL_DIR / "best_currency_model.keras"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        best_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),
]

print("\nStarting training...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Load the checkpoint chosen using validation data.
model = tf.keras.models.load_model(best_path)

print("\nFinal evaluation on held-out test set:")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

print("\nPer-class metrics:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
))

print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
))

with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print(f"\nModel saved to: {best_path}")
print(f"Labels saved to: {MODEL_DIR / 'class_names.json'}")
