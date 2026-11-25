
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# -----------------------------------------
# 1. Paths and configuration
# -----------------------------------------

TEST_DIR = Path(
    r"D:\personal\project\PLANT\tomatoleaf_datasets\test"
)

CHECKPOINT_PATH = Path(
    "checkpoints/best_tomato_model.pth"
)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "independent_test_predictions.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# -----------------------------------------
# 2. Load trained model
# -----------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

class_names = checkpoint["class_names"]
image_size = checkpoint["image_size"]

model = models.mobilenet_v3_small(weights=None)

in_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(
    in_features,
    len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Number of classes:", len(class_names))

# -----------------------------------------
# 3. Image preprocessing
# -----------------------------------------

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------------------
# 4. Find test images
# -----------------------------------------

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

test_images = sorted([
    p for p in TEST_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
])

if not test_images:
    raise RuntimeError("No test images found!")

print("\nIndependent test images:", len(test_images))

# -----------------------------------------
# 5. Run predictions
# -----------------------------------------

results = []

print("\n" + "=" * 75)
print("MEVARG - INDEPENDENT TOMATO IMAGE TEST")
print("=" * 75)

with torch.inference_mode():

    for image_path in test_images:

        try:
            image = Image.open(image_path).convert("RGB")

            input_tensor = transform(image).unsqueeze(0)
            input_tensor = input_tensor.to(DEVICE)

            logits = model(input_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

            top_values, top_indices = torch.topk(
                probabilities,
                k=3
            )

            predicted_index = top_indices[0].item()
            predicted_class = class_names[predicted_index]

            confidence = top_values[0].item() * 100

            # A provisional review flag, not a diagnosis threshold
            review_status = (
                "REVIEW" if confidence < 70
                else "MODEL PREDICTION"
            )

            top3 = []

            for value, index in zip(
                top_values.tolist(),
                top_indices.tolist()
            ):
                top3.append(
                    f"{class_names[index]}: {value * 100:.2f}%"
                )

            print("\nImage:", image_path.name)
            print("Predicted condition:", predicted_class)
            print(f"Confidence: {confidence:.2f}%")
            print("Status:", review_status)

            print("Top 3 predictions:")
            for item in top3:
                print("  ", item)

            results.append({
                "image": image_path.name,
                "predicted_condition": predicted_class,
                "confidence_percent": round(confidence, 2),
                "review_status": review_status,
                "top_2": top3[1],
                "top_3": top3[2]
            })

        except Exception as error:

            print("\nCould not process:", image_path.name)
            print("Error:", error)

# -----------------------------------------
# 6. Save predictions
# -----------------------------------------

if results:

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

print("\n" + "=" * 75)
print("TESTING COMPLETED")
print("Successfully tested:", len(results))
print("Predictions saved to:", OUTPUT_CSV)
print("=" * 75)
