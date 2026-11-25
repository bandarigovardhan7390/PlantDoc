
from pathlib import Path
import csv
import shutil
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from PIL import Image

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

VAL_DIR = PROJECT_DIR / "Plantdoc_Curated" / "validation"

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "models"
    / "plantdoc_classifier"
    / "best_plantdoc_mobilenetv3.pth"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "plantdoc_classifier"
    / "validation_error_audit"
)

MISCLASSIFIED_DIR = OUTPUT_DIR / "misclassified_images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MISCLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224

EXPECTED_CLASSES = [
    "Bacterial_spot",
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Mosaic_virus",
    "Septoria_leaf_spot",
    "Yellow_Leaf_Curl_Virus"
]

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nDevice:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# DATASET
# --------------------------------------------------

if not VAL_DIR.exists():
    raise FileNotFoundError(f"Validation folder not found: {VAL_DIR}")

dataset = datasets.ImageFolder(
    root=str(VAL_DIR),
    transform=transform
)

if dataset.classes != EXPECTED_CLASSES:
    raise ValueError(
        f"Class mismatch!\n"
        f"Expected: {EXPECTED_CLASSES}\n"
        f"Found: {dataset.classes}"
    )

print("Validation images:", len(dataset))
print("Classes:", dataset.classes)

# --------------------------------------------------
# LOAD CHECKPOINT
# --------------------------------------------------

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

if isinstance(checkpoint, dict):
    state_dict = None

    for key in ["model_state_dict", "state_dict", "model"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state_dict = checkpoint[key]
            break

    if state_dict is None:
        # Supports a checkpoint that is itself a state dictionary.
        state_dict = checkpoint
else:
    raise TypeError("Unexpected checkpoint format.")

# Remove DataParallel prefix if present.
state_dict = {
    key.replace("module.", "", 1) if key.startswith("module.") else key: value
    for key, value in state_dict.items()
}

# --------------------------------------------------
# BUILD MODEL
# --------------------------------------------------

model = models.mobilenet_v3_small(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(EXPECTED_CLASSES)
)

model.load_state_dict(state_dict, strict=True)

model = model.to(DEVICE)
model.eval()

print("Checkpoint loaded successfully.")

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------

all_results = []
misclassified_count = 0

print("\nStarting validation audit...\n")

with torch.no_grad():

    for index, (image_tensor, actual_label) in enumerate(dataset):

        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)[0]

        confidence, predicted_label = torch.max(probabilities, dim=0)

        actual_name = EXPECTED_CLASSES[actual_label]
        predicted_name = EXPECTED_CLASSES[predicted_label.item()]

        confidence_value = float(confidence.item()) * 100

        top_values, top_indices = torch.topk(
            probabilities,
            k=min(3, len(EXPECTED_CLASSES))
        )

        top3 = []

        for score, class_index in zip(top_values, top_indices):
            top3.append(
                f"{EXPECTED_CLASSES[class_index.item()]}:"
                f"{score.item() * 100:.2f}%"
            )

        image_path, _ = dataset.samples[index]

        is_correct = actual_label == predicted_label.item()

        result = {
            "image": Path(image_path).name,
            "image_path": str(Path(image_path).resolve()),
            "actual_class": actual_name,
            "predicted_class": predicted_name,
            "confidence_percent": round(confidence_value, 2),
            "correct": is_correct,
            "top_3_predictions": " | ".join(top3)
        }

        all_results.append(result)

        if not is_correct:

            misclassified_count += 1

            # Prefix index prevents filename collisions.
            source = Path(image_path)

            destination = (
                MISCLASSIFIED_DIR
                / f"{index:03d}_{actual_name}_AS_{predicted_name}_{source.name}"
            )

            shutil.copy2(source, destination)

            print(
                f"[ERROR] {source.name}\n"
                f"  Actual:    {actual_name}\n"
                f"  Predicted: {predicted_name}\n"
                f"  Confidence: {confidence_value:.2f}%\n"
            )

# --------------------------------------------------
# SAVE CSV FILES
# --------------------------------------------------

all_csv = OUTPUT_DIR / "all_predictions.csv"
error_csv = OUTPUT_DIR / "misclassified.csv"

fieldnames = [
    "image",
    "image_path",
    "actual_class",
    "predicted_class",
    "confidence_percent",
    "correct",
    "top_3_predictions"
]

with open(all_csv, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)

errors = [row for row in all_results if not row["correct"]]

with open(error_csv, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(errors)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

total = len(all_results)
correct = total - misclassified_count
accuracy = (correct / total) * 100 if total else 0

print("\n" + "=" * 55)
print("VALIDATION ERROR AUDIT COMPLETE")
print("=" * 55)

print("Total images:       ", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", misclassified_count)
print(f"Accuracy:            {accuracy:.2f}%")

print("\nSaved files:")
print("All predictions:", all_csv)
print("Misclassified:  ", error_csv)
print("Error images:   ", MISCLASSIFIED_DIR)

print("\nNext step: inspect the misclassified images and their labels.")
