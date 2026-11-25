"""
Test a trained PlantDoc checkpoint (EfficientNet-B0 or MobileNetV3-Small)
on the images in Plantdoc_Curated/test_external.

Usage (edit the 3 paths below, then run):
    python test_on_external.py
Run it once per model by changing CHECKPOINT.

The model only sees pixels. File names are NEVER used as input.
The folder name is used only to compute "correct / wrong" afterwards.
"""
import hashlib
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ----------------- EDIT THESE -----------------
PROJECT_DIR = Path(r"D:\personal\project\PLANT")
CHECKPOINT = PROJECT_DIR / "models" / "plantdoc_mobilenet_twostage" / "best_plantdoc_mobilenetv3_small_twostage.pth"
# CHECKPOINT = PROJECT_DIR / "models" / "plantdoc_classifier" / "<your mobilenet .pth name>"
# Test folder. For your own confirmed images (e.g. from Google), make a folder with one
# sub-folder per class named like your classes (Bacterial_spot, Early_blight, Healthy, ...)
# and point TEST_DIR at it.
TEST_DIR = PROJECT_DIR / "Plantdoc_Curated" / "test_external"
FIELD_DIR = PROJECT_DIR / "Plantdoc_Curated"   # used only for the duplicate check
OUT_CSV = PROJECT_DIR / "results" / f"external_test_{CHECKPOINT.stem}.csv"
# ----------------------------------------------

# Folder name (lowercase keyword) -> your class. Folders that match nothing
# (e.g. "Tomato leaf", "Tomato mold leaf") are predicted but NOT scored.
FOLDER_KEYWORDS = [
    ("early blight", "Early_blight"),
    ("late blight", "Late_blight"),
    ("bacterial spot", "Bacterial_spot"),
    ("septoria", "Septoria_leaf_spot"),
    ("mosaic", "Mosaic_virus"),
    ("yellow", "Yellow_Leaf_Curl_Virus"),
    ("healthy", "Healthy"),
]

# Stage 1 (tomatoleaf_datasets) has 10 classes. Map them to your 7 classes.
# Leaf_Mold, Spider_mites and Target_Spot have no 7-class equivalent: they stay
# as they are, so an image predicted as one of them counts as wrong.
STAGE1_TO_FIELD = {
    "Tomato___Bacterial_spot": "Bacterial_spot",
    "Tomato___Early_blight": "Early_blight",
    "Tomato___Late_blight": "Late_blight",
    "Tomato___Septoria_leaf_spot": "Septoria_leaf_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus": "Mosaic_virus",
    "Tomato___healthy": "Healthy",
}

def folder_to_class(folder_name):
    n = folder_name.lower().replace("_", " ")
    for key, cls in FOLDER_KEYWORDS:
        if key in n:
            return cls
    return None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CHECKPOINT, map_location=device)
class_names = ckpt["class_names"]
size = ckpt.get("image_size", 288)   # Stage 1 was trained at 288
mean = ckpt.get("mean", [0.485, 0.456, 0.406])
std = ckpt.get("std", [0.229, 0.224, 0.225])
name = str(ckpt.get("model_name", "")).lower()

if "mobile" in name or "model_name" not in ckpt:   # Stage 1 checkpoint has no model_name
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names))
else:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))

model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()
print("Loaded:", ckpt.get("model_name"), "| classes:", class_names)

tf = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()

# exact-duplicate check against train/validation
train_hashes = {}
for split in ("train", "validation"):
    for p in (FIELD_DIR / split).rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            train_hashes[md5(p)] = str(p)

rows = []
for p in sorted(TEST_DIR.rglob("*")):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        continue
    folder = p.parent.name
    truth = folder_to_class(folder)
    img = Image.open(p).convert("RGB")
    with torch.no_grad():
        prob = torch.softmax(model(tf(img).unsqueeze(0).to(device)), dim=1)[0].cpu()
    top = torch.topk(prob, 3)
    pred_raw = class_names[top.indices[0]]
    pred = STAGE1_TO_FIELD.get(pred_raw, pred_raw)
    rows.append({
        "image": p.name,
        "folder": folder,
        "folder_class": truth,
        "predicted": pred,
        "predicted_raw": pred_raw,
        "confidence_%": round(float(top.values[0]) * 100, 2),
        "correct": (pred == truth) if truth else None,
        "top3": " | ".join(f"{class_names[i]}:{float(v)*100:.1f}%" for v, i in zip(top.values, top.indices)),
        "also_in_train_or_val": md5(p) in train_hashes,
    })

df = pd.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

scored = df[df.folder_class.notna()].copy()
scored["correct"] = scored["correct"].astype(bool)
print(f"\nImages: {len(df)} | scored: {len(scored)} | not scored: {len(df)-len(scored)}")
print(f"Exact duplicates of train/val images: {int(df.also_in_train_or_val.sum())}")
if len(scored):
    print(f"Accuracy on scored images: {scored.correct.mean()*100:.2f}% ({int(scored.correct.sum())}/{len(scored)})")
    print("\nPer folder:")
    print(scored.groupby("folder").correct.agg(["sum", "count", "mean"]).round(3))
    print("\nWrong predictions:")
    print(scored[~scored.correct][["image", "folder_class", "predicted", "confidence_%"]].to_string(index=False))
print("\nSaved:", OUT_CSV)