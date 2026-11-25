"""
PLANTDOC - TWO-STAGE MobileNetV3-Small CLASSIFIER

Stage 1: pre-train on tomatoleaf_datasets (clean lab-style images, 10 classes).
Stage 2: fine-tune on Plantdoc_Curated (real field images, 7 classes),
         with strong augmentation, 288px input and label smoothing.

Only Stage 2's validation score matters (Plantdoc_Curated/validation).
The Stage 1 score is on lab-style images, so it will look high and
says little about field performance.

Outputs go to models/plantdoc_mobilenet_twostage and
results/plantdoc_mobilenet_twostage. The checkpoint format works with
test_on_external.py (just point CHECKPOINT at the new .pth file).
"""
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

# ============================================================
# 0. SETTINGS (edit paths if needed)
# ============================================================

PROJECT_DIR = Path(r"D:\personal\project\PLANT")

LAB_DIR = PROJECT_DIR / "tomatoleaf_datasets"
LAB_TRAIN_DIR = LAB_DIR / "train"
LAB_VAL_CANDIDATES = [LAB_DIR / "valid", LAB_DIR / "val", LAB_DIR / "validation"]

FIELD_DIR = PROJECT_DIR / "Plantdoc_Curated"
FIELD_TRAIN_DIR = FIELD_DIR / "train"
FIELD_VAL_DIR = FIELD_DIR / "validation"

MODEL_DIR = PROJECT_DIR / "models" / "plantdoc_mobilenet_twostage"
RESULT_DIR = PROJECT_DIR / "results" / "plantdoc_mobilenet_twostage"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_CKPT = MODEL_DIR / "stage1_lab_pretrain.pth"
FINAL_CKPT = MODEL_DIR / "best_plantdoc_mobilenetv3_small_twostage.pth"

IMAGE_SIZE = 288          # larger than before so small spots stay visible
NUM_WORKERS = 0
SEED = 42

# Stage 1 (lab pre-training)
S1_EPOCHS = 6
S1_BATCH = 32
S1_LR = 5e-4
S1_MAX_PER_CLASS = 700    # cap per class so Stage 1 stays quick
S1_SKIP_IF_EXISTS = True  # reuse stage1 checkpoint if it already exists

# Stage 2 (field fine-tuning)
S2_BATCH = 16
S2_HEAD_EPOCHS = 3        # backbone frozen
S2_FULL_EPOCHS = 30       # whole network, low learning rate
S2_LR_HEAD_ONLY = 1e-3
S2_LR_BACKBONE = 5e-5
S2_LR_HEAD = 3e-4
S2_WEIGHT_DECAY = 1e-2
S2_LABEL_SMOOTHING = 0.1
S2_PATIENCE = 8

EXPECTED_CLASSES = [
    "Bacterial_spot", "Early_blight", "Healthy", "Late_blight",
    "Mosaic_virus", "Septoria_leaf_spot", "Yellow_Leaf_Curl_Virus",
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
use_cuda = device.type == "cuda"
print("=" * 65)
print("PLANTDOC - TWO-STAGE MobileNetV3-Small")
print("=" * 65)
print("Device:", device, "|", torch.cuda.get_device_name(0) if use_cuda else "CPU")

# ============================================================
# 2. TRANSFORMS
# ============================================================

lab_train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.03),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Stronger augmentation to help with messy real-world photos.
field_train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(25),
    transforms.TrivialAugmentWide(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.03),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
])

# ============================================================
# 3. HELPERS
# ============================================================

def build_model(num_classes, pretrained):
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v3_small(weights=weights)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m.to(device)


def make_loader(ds, batch, shuffle):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=use_cuda)


def run_epoch(model, loader, criterion, optimizer=None, scaler=None,
              freeze_backbone_bn=False):
    train = optimizer is not None
    model.train(train)
    if train and freeze_backbone_bn:
        model.features.eval()  # keep BatchNorm statistics fixed while backbone is frozen

    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=use_cuda):
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        n += labels.size(0)
        all_preds += preds.cpu().tolist()
        all_labels += labels.cpu().tolist()
        all_probs += probs.detach().cpu().tolist()

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n, correct / n, macro_f1, all_preds, all_labels, np.array(all_probs)


# ============================================================
# 4. STAGE 1 - PRE-TRAIN ON tomatoleaf_datasets
# ============================================================

if S1_SKIP_IF_EXISTS and STAGE1_CKPT.exists():
    print(f"\nStage 1 checkpoint found, skipping Stage 1: {STAGE1_CKPT}")
    s1 = torch.load(STAGE1_CKPT, map_location=device)
else:
    if not LAB_TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"Lab dataset folder not found: {LAB_TRAIN_DIR}")

    lab_train_full = datasets.ImageFolder(LAB_TRAIN_DIR, transform=lab_train_tf)
    lab_eval_full = datasets.ImageFolder(LAB_TRAIN_DIR, transform=eval_tf)
    lab_classes = lab_train_full.classes
    print("\nStage 1 classes:", lab_classes)

    lab_val_dir = next((d for d in LAB_VAL_CANDIDATES if d.is_dir()), None)
    targets = np.array(lab_train_full.targets)

    if lab_val_dir is not None:
        print("Stage 1 validation folder:", lab_val_dir)
        lab_val_ds = datasets.ImageFolder(lab_val_dir, transform=eval_tf)
        if lab_val_ds.classes != lab_classes:
            raise ValueError("Lab train/validation classes differ.")
        train_idx = np.arange(len(targets))
    else:
        print("No lab validation folder found: using a 10% hold-out split.")
        train_idx, val_idx = train_test_split(
            np.arange(len(targets)), test_size=0.10, stratify=targets, random_state=SEED)
        lab_val_ds = Subset(lab_eval_full, val_idx)

    # cap images per class
    rng = np.random.RandomState(SEED)
    kept = []
    for c in range(len(lab_classes)):
        idx = train_idx[targets[train_idx] == c]
        rng.shuffle(idx)
        kept += idx[:S1_MAX_PER_CLASS].tolist()
    lab_train_ds = Subset(lab_train_full, kept)
    print(f"Stage 1 training images: {len(lab_train_ds)} | validation images: {len(lab_val_ds)}")

    s1_model = build_model(len(lab_classes), pretrained=os.environ.get("NO_PRETRAINED") != "1")
    s1_opt = torch.optim.AdamW(s1_model.parameters(), lr=S1_LR, weight_decay=1e-4)
    s1_sched = torch.optim.lr_scheduler.CosineAnnealingLR(s1_opt, T_max=S1_EPOCHS)
    s1_scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    s1_crit = nn.CrossEntropyLoss()
    s1_train_loader = make_loader(lab_train_ds, S1_BATCH, True)
    s1_val_loader = make_loader(lab_val_ds, S1_BATCH, False)

    for ep in range(S1_EPOCHS):
        tl, ta, _, *_ = run_epoch(s1_model, s1_train_loader, s1_crit, s1_opt, s1_scaler)
        vl, va, vf, *_ = run_epoch(s1_model, s1_val_loader, s1_crit)
        s1_sched.step()
        print(f"[Stage 1] epoch {ep+1}/{S1_EPOCHS} train acc {ta:.3f} | lab val acc {va:.3f} F1 {vf:.3f}")

    s1 = {"model_state_dict": s1_model.state_dict(), "class_names": lab_classes}
    torch.save(s1, STAGE1_CKPT)
    print("Stage 1 checkpoint saved.")

# ============================================================
# 5. STAGE 2 - FINE-TUNE ON Plantdoc_Curated
# ============================================================

for d in (FIELD_TRAIN_DIR, FIELD_VAL_DIR):
    if not d.is_dir():
        raise FileNotFoundError(f"Folder not found: {d}")

train_ds = datasets.ImageFolder(FIELD_TRAIN_DIR, transform=field_train_tf)
val_ds = datasets.ImageFolder(FIELD_VAL_DIR, transform=eval_tf)
class_names = train_ds.classes
num_classes = len(class_names)

if class_names != EXPECTED_CLASSES or val_ds.classes != class_names:
    raise ValueError(f"Unexpected classes. Expected {EXPECTED_CLASSES}, found {class_names}")

print(f"\nStage 2 training images: {len(train_ds)} | validation images: {len(val_ds)}")

counts = np.bincount([y for _, y in train_ds.samples], minlength=num_classes)
w = 1.0 / np.sqrt(counts)
w = w / w.mean()
class_weights = torch.tensor(w, dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=S2_LABEL_SMOOTHING)
val_criterion = nn.CrossEntropyLoss(weight=class_weights)  # no smoothing for evaluation

train_loader = make_loader(train_ds, S2_BATCH, True)
val_loader = make_loader(val_ds, S2_BATCH, False)

# model with the lab-trained backbone, then a fresh 7-class head
model = build_model(len(s1["class_names"]), pretrained=False)
model.load_state_dict(s1["model_state_dict"])
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes).to(device)

scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
history = []
best = {"f1": -1.0, "loss": 1e9, "state": None, "epoch": 0, "acc": 0.0}
bad_epochs = 0


def log_and_track(epoch, stage, tl, ta, vl, va, vf):
    global bad_epochs
    history.append({"epoch": epoch, "stage": stage, "train_loss": tl, "train_accuracy": ta,
                    "val_loss": vl, "val_accuracy": va, "val_macro_f1": vf})
    improved = vf > best["f1"] + 1e-6 or (abs(vf - best["f1"]) <= 1e-6 and vl < best["loss"])
    if improved:
        best.update(f1=vf, loss=vl, acc=va, epoch=epoch, state=copy.deepcopy(model.state_dict()))
        bad_epochs = 0
    else:
        bad_epochs += 1
    print(f"[{stage}] epoch {epoch:2d} | train acc {ta:.3f} | val acc {va:.3f} "
          f"loss {vl:.3f} F1 {vf:.3f}{'  <- best' if improved else ''}")


epoch = 0

# --- 5a. head only ---
for p in model.features.parameters():
    p.requires_grad = False
opt = torch.optim.AdamW(model.classifier.parameters(), lr=S2_LR_HEAD_ONLY, weight_decay=S2_WEIGHT_DECAY)
for _ in range(S2_HEAD_EPOCHS):
    epoch += 1
    tl, ta, *_ = run_epoch(model, train_loader, criterion, opt, scaler, freeze_backbone_bn=True)
    vl, va, vf, *_ = run_epoch(model, val_loader, val_criterion)
    log_and_track(epoch, "head", tl, ta, vl, va, vf)

# --- 5b. full network, low learning rate ---
for p in model.features.parameters():
    p.requires_grad = True
opt = torch.optim.AdamW([
    {"params": model.features.parameters(), "lr": S2_LR_BACKBONE},
    {"params": model.classifier.parameters(), "lr": S2_LR_HEAD},
], weight_decay=S2_WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=S2_FULL_EPOCHS)
bad_epochs = 0
for _ in range(S2_FULL_EPOCHS):
    epoch += 1
    tl, ta, *_ = run_epoch(model, train_loader, criterion, opt, scaler)
    vl, va, vf, *_ = run_epoch(model, val_loader, val_criterion)
    sched.step()
    log_and_track(epoch, "full", tl, ta, vl, va, vf)
    if bad_epochs >= S2_PATIENCE:
        print(f"Early stopping: no improvement for {S2_PATIENCE} epochs.")
        break

# ============================================================
# 6. SAVE BEST MODEL AND REPORTS
# ============================================================

torch.save({
    "model_name": "MobileNetV3-Small",
    "model_state_dict": best["state"],
    "class_names": class_names,
    "image_size": IMAGE_SIZE,
    "mean": MEAN,
    "std": STD,
    "best_epoch": best["epoch"],
    "best_val_accuracy": best["acc"],
    "best_val_macro_f1": best["f1"],
}, FINAL_CKPT)

model.load_state_dict(best["state"])
vl, va, vf, preds, labels, probs = run_epoch(model, val_loader, val_criterion)

report = classification_report(labels, preds, labels=list(range(num_classes)),
                               target_names=class_names, digits=4, zero_division=0)
print("\n" + "=" * 65)
print(f"BEST EPOCH {best['epoch']} | val accuracy {va*100:.2f}% | macro F1 {vf:.4f}")
print("=" * 65)
print(report)

(RESULT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
np.savetxt(RESULT_DIR / "confusion_matrix.csv",
           confusion_matrix(labels, preds, labels=list(range(num_classes))),
           delimiter=",", fmt="%d", header=",".join(class_names), comments="")
(RESULT_DIR / "training_history.json").write_text(json.dumps(history, indent=4), encoding="utf-8")

# per-image predictions (same columns as your earlier audit files)
import csv
rows = []
for (path, true_idx), pred_idx, prob in zip(val_ds.samples, preds, probs):
    top3 = np.argsort(prob)[::-1][:3]
    rows.append({
        "image": Path(path).name,
        "image_path": path,
        "actual_class": class_names[true_idx],
        "predicted_class": class_names[pred_idx],
        "confidence_percent": round(float(prob[pred_idx]) * 100, 2),
        "correct": bool(pred_idx == true_idx),
        "top_3_predictions": " | ".join(f"{class_names[i]}:{prob[i]*100:.2f}%" for i in top3),
    })
for name, subset in (("all_predictions.csv", rows),
                     ("misclassified.csv", [r for r in rows if not r["correct"]])):
    with open(RESULT_DIR / name, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(subset)

print("Model saved:", FINAL_CKPT)
print("Results saved in:", RESULT_DIR)