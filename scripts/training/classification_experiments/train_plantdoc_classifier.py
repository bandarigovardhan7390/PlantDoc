import json
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# PLANTDOC - SEVEN CLASS TOMATO DISEASE CLASSIFIER
# EXPERIMENT B: FRESH IMAGENET-PRETRAINED EFFICIENTNET-B0
# ============================================================

PROJECT_DIR = Path(r"D:\personal\project\PLANT")
DATA_DIR = PROJECT_DIR / "Plantdoc_Curated"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"

MODEL_DIR = PROJECT_DIR / "models" / "plantdoc_efficientnet_b0"
RESULT_DIR = PROJECT_DIR / "results" / "plantdoc_efficientnet_b0"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "best_plantdoc_efficientnet_b0.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 35
HEAD_ONLY_EPOCHS = 4
PARTIAL_FINE_TUNE_EPOCHS = 8

LEARNING_RATE_HEAD = 5e-4
LEARNING_RATE_PARTIAL = 2e-5
LEARNING_RATE_FULL_BACKBONE = 1e-5
LEARNING_RATE_FULL_HEAD = 2e-4

PATIENCE = 9
NUM_WORKERS = 0
SEED = 42

EXPECTED_CLASSES = [
    "Bacterial_spot",
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Mosaic_virus",
    "Septoria_leaf_spot",
    "Yellow_Leaf_Curl_Virus"
]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================================
# 1. REPRODUCIBILITY AND DEVICE
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("PLANTDOC - EFFICIENTNET-B0 FRESH TRAINING EXPERIMENT")
print("=" * 65)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. DATASET VALIDATION
# ============================================================

if not TRAIN_DIR.is_dir():
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

if not VAL_DIR.is_dir():
    raise FileNotFoundError(f"Validation folder not found: {VAL_DIR}")

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.08
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

class_names = train_dataset.classes
num_classes = len(class_names)

if class_names != EXPECTED_CLASSES:
    raise ValueError(
        "Unexpected class order or missing class.\n"
        f"Expected: {EXPECTED_CLASSES}\n"
        f"Found: {class_names}"
    )

if val_dataset.classes != class_names:
    raise ValueError("Training and validation class mappings differ.")

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError("Training and validation class indexes differ.")

if len(train_dataset) != 399 or len(val_dataset) != 98:
    raise ValueError(
        "Dataset counts differ from the verified split. "
        "Check Plantdoc_Curated before training."
    )

print("\nClass mapping:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ============================================================
# 3. CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    [label for _, label in train_dataset.samples],
    minlength=num_classes
)

class_weights_np = 1.0 / np.sqrt(class_counts)
class_weights_np = class_weights_np / class_weights_np.mean()

class_weights = torch.tensor(
    class_weights_np,
    dtype=torch.float32,
    device=device
)

print("\nClass loss weights:")
for name, weight in zip(class_names, class_weights.cpu().tolist()):
    print(f"{name:25s}: {weight:.3f}")


# ============================================================
# 4. DATA LOADERS
# ============================================================

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


# ============================================================
# 5. LOAD A FRESH PRETRAINED EFFICIENTNET-B0
# ============================================================

print("\nLoading fresh ImageNet-pretrained EfficientNet-B0...")

weights = EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device.type == "cuda")
)


# ============================================================
# 6. STAGED UNFREEZING
# ============================================================

def freeze_all_features():
    for parameter in model.features.parameters():
        parameter.requires_grad = False


def unfreeze_last_feature_blocks():
    # EfficientNet-B0 has sequential feature blocks.
    # Unfreeze only the last three blocks for the middle stage.
    for block in list(model.features.children())[-3:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def unfreeze_all_features():
    for parameter in model.features.parameters():
        parameter.requires_grad = True


def make_optimizer(stage):
    if stage == "head":
        return torch.optim.AdamW(
            model.classifier.parameters(),
            lr=LEARNING_RATE_HEAD,
            weight_decay=1e-4
        )

    if stage == "partial":
        backbone_parameters = [
            p for p in model.features.parameters() if p.requires_grad
        ]
        return torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": LEARNING_RATE_PARTIAL},
                {"params": model.classifier.parameters(), "lr": LEARNING_RATE_HEAD}
            ],
            weight_decay=1e-4
        )

    backbone_parameters = [
        p for p in model.features.parameters() if p.requires_grad
    ]
    return torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": LEARNING_RATE_FULL_BACKBONE
            },
            {
                "params": model.classifier.parameters(),
                "lr": LEARNING_RATE_FULL_HEAD
            }
        ],
        weight_decay=1e-4
    )


freeze_all_features()
stage = "head"
optimizer = make_optimizer(stage)

print("Training stage 1: classifier head only")


# ============================================================
# 7. TRAINING AND VALIDATION FUNCTIONS
# ============================================================

def keep_batchnorm_statistics_fixed():
    # Small dataset: avoid noisy BatchNorm running-stat updates.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def train_one_epoch():
    model.train()
    keep_batchnorm_statistics_fixed()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def validate():
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    accuracy = correct / total
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        total_loss / total,
        accuracy,
        macro_f1,
        all_labels,
        all_predictions
    )


# ============================================================
# 8. TRAINING LOOP
# ============================================================

best_val_macro_f1 = -1.0
best_val_accuracy = 0.0
best_model_state = None
epochs_without_improvement = 0
history = []

for epoch in range(EPOCHS):

    # Stage 2: unfreeze only the last three feature blocks.
    if epoch == HEAD_ONLY_EPOCHS:
        print("\nUnfreezing the last three EfficientNet feature blocks...")
        unfreeze_last_feature_blocks()
        stage = "partial"
        optimizer = make_optimizer(stage)

    # Stage 3: unfreeze the complete feature extractor.
    if epoch == HEAD_ONLY_EPOCHS + PARTIAL_FINE_TUNE_EPOCHS:
        print("\nUnfreezing the complete EfficientNet feature extractor...")
        unfreeze_all_features()
        stage = "full"
        optimizer = make_optimizer(stage)

    train_loss, train_acc = train_one_epoch()

    (
        val_loss,
        val_acc,
        val_macro_f1,
        y_true,
        y_pred
    ) = validate()

    print("\n" + "-" * 65)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] | Stage: {stage}")
    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_acc * 100:.2f}%"
    )
    print(
        f"Validation Loss: {val_loss:.4f} | "
        f"Validation Accuracy: {val_acc * 100:.2f}%"
    )
    print(f"Validation Macro F1: {val_macro_f1:.4f}")

    history.append({
        "epoch": epoch + 1,
        "stage": stage,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_macro_f1
    })

    # Save the best model by validation macro F1.
    if val_macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = val_macro_f1
        best_val_accuracy = val_acc
        best_model_state = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "model_name": "EfficientNet-B0",
                "model_state_dict": best_model_state,
                "class_names": class_names,
                "image_size": IMAGE_SIZE,
                "mean": MEAN,
                "std": STD,
                "best_epoch": epoch + 1,
                "best_val_accuracy": best_val_accuracy,
                "best_val_macro_f1": best_val_macro_f1
            },
            CHECKPOINT_PATH
        )

        print("Best EfficientNet-B0 checkpoint saved.")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# ============================================================
# 9. FINAL VALIDATION WITH BEST CHECKPOINT
# ============================================================

if best_model_state is None:
    raise RuntimeError("Training did not produce a valid checkpoint.")

model.load_state_dict(best_model_state)

(
    val_loss,
    val_acc,
    val_macro_f1,
    y_true,
    y_pred
) = validate()

print("\n" + "=" * 65)
print("EFFICIENTNET-B0 FINAL VALIDATION RESULTS")
print("=" * 65)
print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
print(f"Final validation accuracy: {val_acc * 100:.2f}%")

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(num_classes)),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\nCLASSIFICATION REPORT\n")
print(report)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(num_classes))
)

print("\nCONFUSION MATRIX\n")
print(cm)


# ============================================================
# 10. SAVE EXPERIMENT RESULTS
# ============================================================

with open(
    RESULT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

np.savetxt(
    RESULT_DIR / "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

with open(
    RESULT_DIR / "training_history.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(history, file, indent=4)

summary = {
    "model": "EfficientNet-B0",
    "initialization": "ImageNet pretrained; fresh classifier head",
    "dataset": "Plantdoc_Curated",
    "classes": class_names,
    "training_images": len(train_dataset),
    "validation_images": len(val_dataset),
    "best_epoch": int(
        torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)["best_epoch"]
    ),
    "best_validation_accuracy": best_val_accuracy,
    "best_validation_macro_f1": best_val_macro_f1,
    "image_size": IMAGE_SIZE,
    "external_test_images_used_for_training": False
}

with open(
    RESULT_DIR / "training_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

print("\nTRAINING COMPLETED")
print("Checkpoint:", CHECKPOINT_PATH)
print("Results:", RESULT_DIR)
print("\nExternal test images were NOT used during training.")
