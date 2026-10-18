
from pathlib import Path
import random
import shutil

# Original downloaded dataset
SOURCE = (
    Path.home()
    / "Downloads"
    / "archive (3)"
    / "Indian currency dataset v1"
)

# Destination expected by train_model.py
DEST = Path(__file__).resolve().parent / "data" / "currency"

SEED = 42
TEST_RATIO = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)

train_source = SOURCE / "training"
val_source = SOURCE / "validation"

if not train_source.is_dir() or not val_source.is_dir():
    raise FileNotFoundError(
        f"Could not find training/validation folders in {SOURCE}"
    )

# Never overwrite an existing prepared dataset.
if DEST.exists() and any(DEST.iterdir()):
    raise FileExistsError(
        f"{DEST} already contains files. "
        "Rename or back it up before preparing again."
    )

train_classes = {
    p.name for p in train_source.iterdir() if p.is_dir()
}
val_classes = {
    p.name for p in val_source.iterdir() if p.is_dir()
}

if train_classes != val_classes:
    raise ValueError(
        f"Class mismatch!\nTraining: {sorted(train_classes)}"
        f"\nValidation: {sorted(val_classes)}"
    )

if len(train_classes) < 2:
    raise ValueError("At least two classes are required.")

for split in ("train", "val", "test"):
    for class_name in sorted(train_classes):
        (DEST / split / class_name).mkdir(
            parents=True, exist_ok=True
        )

# Keep the provided validation images separate.
for class_name in sorted(val_classes):
    files = [
        p for p in (val_source / class_name).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not files:
        raise ValueError(
            f"No supported images in validation/{class_name}"
        )

    for src in files:
        shutil.copy2(
            src,
            DEST / "val" / class_name / src.name
        )

# Split each labelled training class into train and test.
for class_name in sorted(train_classes):
    files = [
        p for p in (train_source / class_name).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(files) < 10:
        raise ValueError(
            f"Too few images in training/{class_name}: {len(files)}"
        )

    random.shuffle(files)
    test_count = max(1, round(len(files) * TEST_RATIO))
    test_files = files[:test_count]
    train_files = files[test_count:]

    for split, selected in (
        ("train", train_files),
        ("test", test_files),
    ):
        for src in selected:
            shutil.copy2(
                src,
                DEST / split / class_name / src.name
            )

print("\nDataset prepared successfully.")
print(f"Destination: {DEST}")

for split in ("train", "val", "test"):
    print(f"\n{split.upper()}")
    for class_name in sorted(train_classes):
        count = sum(
            1 for p in (DEST / split / class_name).iterdir()
            if p.is_file()
        )
        print(f"  {class_name}: {count} images")
