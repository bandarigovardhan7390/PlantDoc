"""
Run prediction-only inference with trained EfficientNet-B0 checkpoints
on an external image folder.

- Reads images directly inside EXTERNAL_DIR and in any nested subfolders.
- Does not require class-labelled subfolders.
- Evaluates each seed checkpoint and averages probabilities for an ensemble.
- Saves per-image predictions, confidence scores, top-3 classes, and duplicate flags.
- Does NOT calculate accuracy/F1 because ground-truth labels are not supplied.
- Nothing is trained or changed.

Outputs: results/plantdoc_final/external/
"""

import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


# ------------------------- SETTINGS -------------------------
PROJECT_DIR = Path(r"D:\personal\project\PLANT")
EXTERNAL_DIR = PROJECT_DIR / "tomatoleaf_datasets" / "test1"
TRAIN_VAL_DIR = PROJECT_DIR / "Plantdoc_Curated_v2"
MODEL_DIR = PROJECT_DIR / "models" / "plantdoc_final"
OUT_DIR = PROJECT_DIR / "results" / "plantdoc_final" / "external"

CHECKPOINTS = [MODEL_DIR / f"effb0_imagenet_seed{s}_last.pth" for s in (42, 1, 7)]

CLASSES = [
    "Bacterial_spot",
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Mosaic_virus",
    "Septoria_leaf_spot",
    "Yellow_Leaf_Curl_Virus",
]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH_SIZE = 32
# ------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    names = ck["class_names"]
    if names != CLASSES:
        raise ValueError(f"{path.name}: class order differs: {names}")

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(names))
    model.load_state_dict(ck["model_state_dict"])

    tf = transforms.Compose([
        transforms.Resize((ck["image_size"], ck["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ck["mean"], ck["std"]),
    ])
    return model.to(device).eval(), tf


def main():
    if not EXTERNAL_DIR.is_dir():
        raise FileNotFoundError(f"External folder not found: {EXTERNAL_DIR}")

    missing = [str(p) for p in CHECKPOINTS if not p.is_file()]
    if missing:
        raise FileNotFoundError("Checkpoint(s) not found:\n" + "\n".join(missing))

    # Include images in test1 itself and in all nested folders.
    image_files = sorted(
        p for p in EXTERNAL_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    )

    # Verify that files are readable images before inference.
    items, unreadable = [], []
    for path in image_files:
        try:
            with Image.open(path) as im:
                im.verify()
            items.append(path)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            unreadable.append((path, str(exc)))

    print(f"Image files found: {len(image_files)}")
    print(f"Readable images used: {len(items)}")
    if unreadable:
        print(f"Unreadable/invalid images skipped: {len(unreadable)}")
        for path, error in unreadable:
            print(f"  {path}: {error}")

    if not items:
        raise RuntimeError(
            f"No readable images found in {EXTERNAL_DIR}. "
            f"Supported extensions: {sorted(EXTS)}"
        )

    # Exact-duplicate check against the existing train/validation folders.
    seen = set()
    if TRAIN_VAL_DIR.is_dir():
        for path in TRAIN_VAL_DIR.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in EXTS
                and path.parent.parent.name in ("train", "validation")
            ):
                seen.add(md5(path))
    else:
        print(f"Warning: train/validation directory not found: {TRAIN_VAL_DIR}")

    duplicate_flags = [md5(path) in seen for path in items]
    print(
        "Exact duplicates of train/validation images: "
        f"{sum(duplicate_flags)} of {len(items)}"
    )

    # Run each seed checkpoint.
    probabilities_per_model = []
    for checkpoint in CHECKPOINTS:
        print(f"\nLoading checkpoint: {checkpoint.name}")
        model, tf = load_model(checkpoint)
        model_probs = []

        with torch.no_grad():
            for start in range(0, len(items), BATCH_SIZE):
                batch_paths = items[start:start + BATCH_SIZE]
                tensors = []
                for path in batch_paths:
                    with Image.open(path) as image:
                        tensors.append(tf(image.convert("RGB")))

                batch = torch.stack(tensors).to(device)
                logits = model(batch).float()
                model_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

        probabilities_per_model.append(np.concatenate(model_probs, axis=0))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average class probabilities across the three seed models.
    ensemble_probs = np.mean(probabilities_per_model, axis=0)
    ensemble_predictions = ensemble_probs.argmax(axis=1)

    seed_predictions = [p.argmax(axis=1) for p in probabilities_per_model]

    output_csv = OUT_DIR / "external_predictions.csv"
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        header = [
            "image",
            "relative_path",
            "ensemble_predicted_class",
            "ensemble_confidence_percent",
            "top_3",
            "duplicate_of_train_val",
        ]
        for seed in (42, 1, 7):
            header.extend([f"seed_{seed}_predicted_class", f"seed_{seed}_confidence_percent"])
        writer.writerow(header)

        for index, path in enumerate(items):
            probs = ensemble_probs[index]
            predicted_index = int(ensemble_predictions[index])
            top3_indices = np.argsort(probs)[::-1][:3]
            top3 = " | ".join(
                f"{CLASSES[i]}:{probs[i] * 100:.2f}%"
                for i in top3_indices
            )

            row = [
                path.name,
                str(path.relative_to(EXTERNAL_DIR)),
                CLASSES[predicted_index],
                round(float(probs[predicted_index]) * 100, 2),
                top3,
                duplicate_flags[index],
            ]

            for model_probs, model_preds in zip(
                probabilities_per_model, seed_predictions
            ):
                seed_pred = int(model_preds[index])
                row.extend([
                    CLASSES[seed_pred],
                    round(float(model_probs[index, seed_pred]) * 100, 2),
                ])

            writer.writerow(row)

    # Save a readable summary. No accuracy/F1 is reported without true labels.
    predicted_counts = Counter(CLASSES[int(i)] for i in ensemble_predictions)
    summary_path = OUT_DIR / "prediction_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("EXTERNAL IMAGE PREDICTION SUMMARY\n")
        fh.write("=" * 40 + "\n")
        fh.write(f"External directory: {EXTERNAL_DIR}\n")
        fh.write(f"Readable images predicted: {len(items)}\n")
        fh.write(f"Exact train/validation duplicates: {sum(duplicate_flags)}\n")
        fh.write("\nEnsemble predicted-class counts:\n")
        for class_name in CLASSES:
            fh.write(f"{class_name}: {predicted_counts.get(class_name, 0)}\n")
        fh.write(
            "\nAccuracy and F1-score were not calculated because "
            "ground-truth labels were not provided.\n"
        )

    print("\n" + "=" * 60)
    print("ENSEMBLE PREDICTION COUNTS")
    print("=" * 60)
    for class_name in CLASSES:
        print(f"{class_name:28s}: {predicted_counts.get(class_name, 0)}")

    print("\nPrediction-only evaluation completed.")
    print("Predictions CSV:", output_csv)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
