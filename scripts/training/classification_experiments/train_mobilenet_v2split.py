"""
PLANTDOC - MobileNetV3-Small baseline on the folder-based split (Plantdoc_Curated_v2)

Single-stage transfer learning (ImageNet weights, whole network trained), 224px,
class-weighted loss, mixed precision, best checkpoint chosen by validation macro F1.

Because the validation set is only ~100 images, one run can swing by several
points. So this script trains with 3 different seeds and reports mean +/- std.
It also reports the LAST-epoch score and the average of the last 5 epochs,
because "best epoch on validation" is slightly optimistic (it is chosen on the
same images it is scored on).

Optional: DROP_CONFLICT_PAIRS = True removes the images that audit/near_duplicates.csv
shows sitting in two different class folders (same photo, two labels).

Outputs: models/plantdoc_mobilenet_v2split and results/plantdoc_mobilenet_v2split
Checkpoints work with test_on_external.py.
"""
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

# ------------------------- SETTINGS -------------------------
PROJECT_DIR = Path(r"D:\personal\project\PLANT")
DATA_DIR = PROJECT_DIR / "Plantdoc_Curated_v2"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
OUT_MODELS = PROJECT_DIR / "models" / "plantdoc_mobilenet_v2split"
OUT_RESULTS = PROJECT_DIR / "results" / "plantdoc_mobilenet_v2split"

SEEDS = [42, 1, 7]
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
DROP_CONFLICT_PAIRS = False   # set True to remove same-photo-in-two-classes images

# folder name used in your original data -> class name (same mapping as the split script)
FOLDER_TO_CLASS = {
    "bactyerial spot": "Bacterial_spot", "bacterial spot": "Bacterial_spot",
    "early blight": "Early_blight", "healthy": "Healthy", "late blight": "Late_blight",
    "mosaic virus": "Mosaic_virus", "septoria spot": "Septoria_leaf_spot",
    "yellow virus": "Yellow_Leaf_Curl_Virus",
}
EXPECTED_CLASSES = [
    "Bacterial_spot", "Early_blight", "Healthy", "Late_blight",
    "Mosaic_virus", "Septoria_leaf_spot", "Yellow_Leaf_Curl_Virus",
]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# ------------------------------------------------------------

OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_RESULTS.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"
print("Device:", device, "|", torch.cuda.get_device_name(0) if use_cuda else "CPU")

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def conflict_set():
    """(class, file name) of every image that sits in two different class folders."""
    f = DATA_DIR / "audit" / "near_duplicates.csv"
    if not f.exists():
        return set()
    df = pd.read_csv(f, encoding="utf-8-sig")
    df = df[df["CLASS_CONFLICT"] == True]
    out = set()
    for _, r in df.iterrows():
        for side in ("a", "b"):
            cls = FOLDER_TO_CLASS.get(str(r[f"{side}_folder"]).strip().lower())
            out.add((cls, r[f"{side}_file"]))
    return out


def filter_samples(ds, drop):
    ds.samples = [s for s in ds.samples if (ds.classes[s[1]], Path(s[0]).name) not in drop]
    ds.imgs = ds.samples
    ds.targets = [s[1] for s in ds.samples]


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total, correct, n = 0.0, 0, 0
    preds_all, labels_all, probs_all = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=use_cuda):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        p = torch.softmax(logits.float(), dim=1)
        pred = p.argmax(1)
        total += loss.item() * y.size(0)
        correct += (pred == y).sum().item()
        n += y.size(0)
        preds_all += pred.cpu().tolist()
        labels_all += y.cpu().tolist()
        probs_all += p.detach().cpu().tolist()
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total / n, correct / n, f1, preds_all, labels_all, np.array(probs_all)


def train_one(seed, drop):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tf)
    if train_ds.classes != EXPECTED_CLASSES or val_ds.classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected classes: {train_ds.classes}")
    if drop:
        filter_samples(train_ds, drop)
        filter_samples(val_ds, drop)
    classes = train_ds.classes
    k = len(classes)
    print(f"\n=== seed {seed} | train {len(train_ds)} | validation {len(val_ds)} ===")

    counts = np.bincount([y for _, y in train_ds.samples], minlength=k)
    w = 1.0 / np.sqrt(counts)
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g,
                              num_workers=NUM_WORKERS, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=use_cuda)

    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, k)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    hist = []
    best = {"f1": -1.0, "loss": 1e9, "epoch": 0, "state": None}
    for ep in range(1, EPOCHS + 1):
        tl, ta, *_ = run_epoch(model, train_loader, criterion, opt, scaler)
        vl, va, vf, *_ = run_epoch(model, val_loader, criterion)
        sched.step()
        hist.append({"epoch": ep, "train_loss": tl, "train_accuracy": ta,
                     "val_loss": vl, "val_accuracy": va, "val_macro_f1": vf})
        better = vf > best["f1"] + 1e-6 or (abs(vf - best["f1"]) <= 1e-6 and vl < best["loss"])
        if better:
            best.update(f1=vf, loss=vl, epoch=ep, state=copy.deepcopy(model.state_dict()))
        print(f"epoch {ep:2d} | train acc {ta:.3f} | val acc {va:.3f} loss {vl:.3f} F1 {vf:.3f}"
              f"{'  <- best' if better else ''}")

    # save best checkpoint + reports
    model.load_state_dict(best["state"])
    vl, va, vf, preds, labels, probs = run_epoch(model, val_loader, criterion)
    out = OUT_RESULTS / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": "MobileNetV3-Small", "model_state_dict": best["state"],
                "class_names": classes, "image_size": IMAGE_SIZE, "mean": MEAN, "std": STD,
                "best_epoch": best["epoch"], "best_val_macro_f1": vf, "seed": seed},
               OUT_MODELS / f"best_mobilenetv3_small_v2split_seed{seed}.pth")
    report = classification_report(labels, preds, labels=list(range(k)), target_names=classes,
                                   digits=4, zero_division=0)
    (out / "classification_report.txt").write_text(report, encoding="utf-8")
    np.savetxt(out / "confusion_matrix.csv", confusion_matrix(labels, preds, labels=list(range(k))),
               delimiter=",", fmt="%d", header=",".join(classes), comments="")
    (out / "training_history.json").write_text(json.dumps(hist, indent=4), encoding="utf-8")
    rows = []
    for (path, t), pr, pb in zip(val_ds.samples, preds, probs):
        top3 = np.argsort(pb)[::-1][:3]
        rows.append({"image": Path(path).name, "image_path": path, "actual_class": classes[t],
                     "predicted_class": classes[pr], "confidence_percent": round(float(pb[pr]) * 100, 2),
                     "correct": bool(pr == t),
                     "top_3_predictions": " | ".join(f"{classes[i]}:{pb[i]*100:.2f}%" for i in top3)})
    for name, sub in (("all_predictions.csv", rows), ("misclassified.csv", [r for r in rows if not r["correct"]])):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(sub)
    print(report)

    last5 = hist[-5:]
    return {"seed": seed, "best_epoch": best["epoch"],
            "best_acc": va, "best_f1": vf,
            "last_acc": hist[-1]["val_accuracy"], "last_f1": hist[-1]["val_macro_f1"],
            "last5_mean_acc": float(np.mean([h["val_accuracy"] for h in last5])),
            "last5_mean_f1": float(np.mean([h["val_macro_f1"] for h in last5])),
            "val_images": len(val_ds)}


if __name__ == "__main__":
    drop = conflict_set() if DROP_CONFLICT_PAIRS else set()
    if DROP_CONFLICT_PAIRS:
        print("Dropping", len(drop), "images that sit in two class folders")
    results = [train_one(s, drop) for s in SEEDS]
    df = pd.DataFrame(results)
    df.to_csv(OUT_RESULTS / "summary_by_seed.csv", index=False)

    def ms(col, pct=False):
        m, s = df[col].mean(), df[col].std(ddof=0)
        return f"{m*100:.1f}% +/- {s*100:.1f}" if pct else f"{m:.3f} +/- {s:.3f}"

    print("\n" + "=" * 62)
    print(f"SUMMARY over seeds {SEEDS}  (validation images: {df.val_images.iloc[0]})")
    print("=" * 62)
    print(df.round(4).to_string(index=False))
    print()
    print("Best epoch    accuracy:", ms("best_acc", True), "| macro F1:", ms("best_f1"))
    print("Last epoch    accuracy:", ms("last_acc", True), "| macro F1:", ms("last_f1"))
    print("Last-5 mean   accuracy:", ms("last5_mean_acc", True), "| macro F1:", ms("last5_mean_f1"))
    print("\nCompare with your earlier plain MobileNet: 65.3% accuracy, macro F1 0.633 (one run).")