
import json
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path(
    r"D:\personal\project\PLANT\tomatoleaf_datasets"
)

OUTPUT_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
HEAD_ONLY_EPOCHS = 3

LEARNING_RATE_HEAD = 0.001
LEARNING_RATE_FINE = 0.0001

PATIENCE = 7
NUM_WORKERS = 0

SEED = 42

# --------------------------------------------------
# 2. Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------
# 3. Image transformations
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# 4. Load datasets
# --------------------------------------------------

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_transform
)

class_names = train_dataset.classes
num_classes = len(class_names)

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError(
        "Training and validation class mappings differ!"
    )

if num_classes != 10:
    raise ValueError(
        f"Expected 10 classes, found {num_classes}"
    )

print("\nClasses:")
for idx, name in enumerate(class_names):
    print(idx, ":", name)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

# --------------------------------------------------
# 5. Load MobileNetV3-Small
# --------------------------------------------------

weights = MobileNet_V3_Small_Weights.DEFAULT

model = models.mobilenet_v3_small(
    weights=weights
)

# Replace final classification layer
in_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(
    in_features,
    num_classes
)

model = model.to(device)

# --------------------------------------------------
# 6. Freeze feature extractor initially
# --------------------------------------------------

for param in model.features.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE_HEAD,
    weight_decay=1e-4
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device.type == "cuda")
)

# --------------------------------------------------
# 7. Training and validation functions
# --------------------------------------------------

def train_one_epoch():

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda")
        ):

            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return total_loss / total, correct / total


def validate():

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):

                outputs = model(images)
                loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    return (
        total_loss / total,
        correct / total,
        all_labels,
        all_predictions
    )


# --------------------------------------------------
# 8. Main training loop
# --------------------------------------------------

best_val_accuracy = 0.0
best_val_loss = float("inf")
epochs_without_improvement = 0

best_model_state = None

history = []

for epoch in range(EPOCHS):

    # Unfreeze the complete network after initial head training
    if epoch == HEAD_ONLY_EPOCHS:

        print("\nUnfreezing feature extractor...")

        for param in model.features.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE_FINE,
            weight_decay=1e-4
        )

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc, y_true, y_pred = validate()

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_acc * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_acc * 100:.2f}%"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    # Save the best validation checkpoint
    if val_acc > best_val_accuracy:

        best_val_accuracy = val_acc
        best_val_loss = val_loss

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": best_model_state,
                "class_names": class_names,
                "image_size": IMAGE_SIZE,
                "best_val_accuracy": best_val_accuracy
            },
            CHECKPOINT_DIR / "best_tomato_model.pth"
        )

        print("Best model checkpoint saved.")

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    # Early stopping
    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# --------------------------------------------------
# 9. Final validation evaluation
# --------------------------------------------------

if best_model_state is not None:

    model.load_state_dict(best_model_state)

val_loss, val_acc, y_true, y_pred = validate()

print("\nFINAL VALIDATION RESULTS")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Final validation accuracy: "
    f"{val_acc * 100:.2f}%"
)

print("\nClassification Report:\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

print("\nConfusion Matrix:\n")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)

# --------------------------------------------------
# 10. Save evaluation outputs
# --------------------------------------------------

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

np.savetxt(
    OUTPUT_DIR / "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

with open(
    OUTPUT_DIR / "training_history.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(history, f, indent=4)

print("\nTraining completed.")
print("Best model:", CHECKPOINT_DIR / "best_tomato_model.pth")
print("Results folder:", OUTPUT_DIR)
