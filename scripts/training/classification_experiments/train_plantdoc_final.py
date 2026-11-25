"""
PLANTDOC - EfficientNet-B0, staged fine-tuning, 3 seeds, optional lab pre-training

Recipe (taken from your best-scoring staged script):
  - ImageNet-pretrained EfficientNet-B0, fresh 7-class head
  - head only -> last 3 blocks -> whole network, low learning rates, BatchNorm kept fixed
  - class-weighted loss, mixed precision

What is new:
  - Dataset: Plantdoc_Curated_v2 (folder-based split, near-duplicates kept on one side)
  - 3 seeds, fixed number of epochs (no early stopping), so best / last / last-5 scores are comparable
  - MODES: "imagenet"      = start from ImageNet only
           "lab_pretrain"  = first train on tomatoleaf_datasets (clean single leaves),
                             then continue on Plantdoc. Run both to see if lab data really helps.
  - DROP_CONFLICT_PAIRS = True removes the 14 images that sit in two class folders
  - Headline number = last-5-epoch average (best epoch is chosen on the same
    validation images, so it is optimistic)

Outputs: models/plantdoc_final and results/plantdoc_final
Checkpoints use the same keys as before, so test_on_external.py can load them.
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights

# ------------------------- SETTINGS -------------------------
PROJECT_DIR = Path(r"D:\personal\project\PLANT")
DATA_DIR = PROJECT_DIR / "Plantdoc_Curated_v2"
TRAIN_DIR, VAL_DIR = DATA_DIR / "train", DATA_DIR / "validation"
LAB_DIR = PROJECT_DIR / "tomatoleaf_datasets"
LAB_TRAIN_DIR = LAB_DIR / "train"
LAB_VAL_CANDIDATES = [LAB_DIR / "val", LAB_DIR / "valid", LAB_DIR / "validation"]
OUT_MODELS = PROJECT_DIR / "models" / "plantdoc_final"
OUT_RESULTS = PROJECT_DIR / "results" / "plantdoc_final"
LAB_CKPT = OUT_MODELS / "stage1_lab_backbone.pth"

MODES = ["imagenet", "lab_pretrain"]
SEEDS = [42, 1, 7]
DROP_CONFLICT_PAIRS = True

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
EPOCHS = 30
HEAD_ONLY_EPOCHS = 4
PARTIAL_EPOCHS = 8            # then the rest is full-network training
LR_HEAD_ONLY = 5e-4
LR_PARTIAL_BACKBONE, LR_PARTIAL_HEAD = 2e-5, 5e-4
LR_FULL_BACKBONE, LR_FULL_HEAD = 1e-5, 2e-4
WEIGHT_DECAY = 1e-4
STRONG_AUG = True             # random crops + rotation; set False for the mild original

# lab pre-training
LAB_EPOCHS = 4
LAB_BATCH = 32
LAB_LR = 3e-4
LAB_MAX_PER_CLASS = 700

FOLDER_TO_CLASS = {
    "bactyerial spot": "Bacterial_spot", "bacterial spot": "Bacterial_spot",
    "early blight": "Early_blight", "healthy": "Healthy", "late blight": "Late_blight",
    "mosaic virus": "Mosaic_virus", "septoria spot": "Septoria_leaf_spot",
    "yellow virus": "Yellow_Leaf_Curl_Virus",
}
EXPECTED_CLASSES = ["Bacterial_spot", "Early_blight", "Healthy", "Late_blight",
                    "Mosaic_virus", "Septoria_leaf_spot", "Yellow_Leaf_Curl_Virus"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# ------------------------------------------------------------

OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_RESULTS.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"
print("Device:", device, "|", torch.cuda.get_device_name(0) if use_cuda else "CPU")

eval_tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
if STRONG_AUG:
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(0.15, 0.15, 0.1), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
else:
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), transforms.ColorJitter(0.1, 0.1, 0.08),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def conflict_set():
    """(class, file name) of every image that sits in two different class folders."""
    f = DATA_DIR / "audit" / "near_duplicates.csv"
    if not f.exists():
        print("WARNING: audit/near_duplicates.csv not found, nothing dropped")
        return set()
    df = pd.read_csv(f, encoding="utf-8-sig")
    df = df[df["CLASS_CONFLICT"] == True]
    out = set()
    for _, r in df.iterrows():
        for side in ("a", "b"):
            out.add((FOLDER_TO_CLASS.get(str(r[f"{side}_folder"]).strip().lower()), r[f"{side}_file"]))
    return out


def filter_samples(ds, drop):
    ds.samples = [s for s in ds.samples if (ds.classes[s[1]], Path(s[0]).name) not in drop]
    ds.imgs = ds.samples
    ds.targets = [s[1] for s in ds.samples]


def build_model(k):
    m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, k)
    return m.to(device)


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, freeze_bn=False):
    train = optimizer is not None
    model.train(train)
    if train and freeze_bn:  # small data: keep BatchNorm statistics fixed
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
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


# ---------------------- Stage 1: lab pre-training ----------------------
def lab_pretrain():
    if LAB_CKPT.exists():
        print("Lab backbone found, reusing:", LAB_CKPT)
        return torch.load(LAB_CKPT, map_location=device)
    if not LAB_TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"Lab dataset not found: {LAB_TRAIN_DIR}")
    set_seed(42)
    full = datasets.ImageFolder(LAB_TRAIN_DIR, transform=train_tf)
    lab_val_dir = next((d for d in LAB_VAL_CANDIDATES if d.is_dir()), None)
    if lab_val_dir is None:
        raise FileNotFoundError("No lab validation folder (val / valid / validation)")
    val = datasets.ImageFolder(lab_val_dir, transform=eval_tf)
    if val.classes != full.classes:
        raise ValueError("Lab train/validation classes differ")
    targets = np.array(full.targets)
    rng = np.random.RandomState(42)
    kept = []
    for c in range(len(full.classes)):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        kept += idx[:LAB_MAX_PER_CLASS].tolist()
    train_ds = Subset(full, kept)
    print(f"Stage 1 | classes {full.classes} | train {len(train_ds)} | val {len(val)}")
    model = build_model(len(full.classes))
    opt = torch.optim.AdamW(model.parameters(), lr=LAB_LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=LAB_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    crit = nn.CrossEntropyLoss()
    tl_ = DataLoader(train_ds, batch_size=LAB_BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=use_cuda)
    vl_ = DataLoader(val, batch_size=LAB_BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=use_cuda)
    for ep in range(1, LAB_EPOCHS + 1):
        _, ta, *_ = run_epoch(model, tl_, crit, opt, scaler)
        _, va, vf, *_ = run_epoch(model, vl_, crit)
        sched.step()
        print(f"[Stage 1] epoch {ep}/{LAB_EPOCHS} train acc {ta:.3f} | lab val acc {va:.3f} F1 {vf:.3f}")
    ck = {"features": model.features.state_dict(), "lab_classes": full.classes}
    torch.save(ck, LAB_CKPT)
    return ck


# ---------------------- Stage 2: Plantdoc fine-tuning ----------------------
def train_field(mode, seed, drop, lab_ck):
    set_seed(seed)
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tf)
    if train_ds.classes != EXPECTED_CLASSES or val_ds.classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected classes: {train_ds.classes}")
    if drop:
        filter_samples(train_ds, drop)
        filter_samples(val_ds, drop)
    classes, k = train_ds.classes, len(train_ds.classes)
    print(f"\n=== {mode} | seed {seed} | train {len(train_ds)} | validation {len(val_ds)} ===")

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

    model = build_model(k)
    if mode == "lab_pretrain":
        model.features.load_state_dict(lab_ck["features"])

    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    for p in model.features.parameters():
        p.requires_grad = False
    stage = "head"
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=LR_HEAD_ONLY, weight_decay=WEIGHT_DECAY)

    hist = []
    best = {"f1": -1.0, "loss": 1e9, "epoch": 0, "state": None}
    for ep in range(1, EPOCHS + 1):
        if ep == HEAD_ONLY_EPOCHS + 1:
            for blk in list(model.features.children())[-3:]:
                for p in blk.parameters():
                    p.requires_grad = True
            stage = "partial"
            opt = torch.optim.AdamW(
                [{"params": [p for p in model.features.parameters() if p.requires_grad], "lr": LR_PARTIAL_BACKBONE},
                 {"params": model.classifier.parameters(), "lr": LR_PARTIAL_HEAD}], weight_decay=WEIGHT_DECAY)
        if ep == HEAD_ONLY_EPOCHS + PARTIAL_EPOCHS + 1:
            for p in model.features.parameters():
                p.requires_grad = True
            stage = "full"
            opt = torch.optim.AdamW(
                [{"params": model.features.parameters(), "lr": LR_FULL_BACKBONE},
                 {"params": model.classifier.parameters(), "lr": LR_FULL_HEAD}], weight_decay=WEIGHT_DECAY)
        tl, ta, *_ = run_epoch(model, train_loader, criterion, opt, scaler, freeze_bn=True)
        vl, va, vf, *_ = run_epoch(model, val_loader, criterion)
        hist.append({"epoch": ep, "stage": stage, "train_loss": tl, "train_accuracy": ta,
                     "val_loss": vl, "val_accuracy": va, "val_macro_f1": vf})
        better = vf > best["f1"] + 1e-6 or (abs(vf - best["f1"]) <= 1e-6 and vl < best["loss"])
        if better:
            best.update(f1=vf, loss=vl, epoch=ep, state=copy.deepcopy(model.state_dict()))
        print(f"[{stage:7s}] epoch {ep:2d} | train acc {ta:.3f} | val acc {va:.3f} loss {vl:.3f} "
              f"F1 {vf:.3f}{'  <- best' if better else ''}")

    last_state = copy.deepcopy(model.state_dict())
    meta = {"model_name": "EfficientNet-B0", "class_names": classes, "image_size": IMAGE_SIZE,
            "mean": MEAN, "std": STD, "seed": seed, "mode": mode}
    tag = f"{mode}_seed{seed}"
    torch.save({**meta, "model_state_dict": best["state"], "best_epoch": best["epoch"]},
               OUT_MODELS / f"effb0_{tag}_best.pth")
    torch.save({**meta, "model_state_dict": last_state, "best_epoch": EPOCHS},
               OUT_MODELS / f"effb0_{tag}_last.pth")

    # reports for the LAST epoch model (not picked on validation)
    vl, va, vf, preds, labels, probs = run_epoch(model, val_loader, criterion)
    out = OUT_RESULTS / mode / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    report = classification_report(labels, preds, labels=list(range(k)), target_names=classes,
                                   digits=4, zero_division=0)
    (out / "classification_report_last_epoch.txt").write_text(report, encoding="utf-8")
    np.savetxt(out / "confusion_matrix_last_epoch.csv",
               confusion_matrix(labels, preds, labels=list(range(k))),
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
    print("Last-epoch report:\n" + report)

    last5 = hist[-5:]
    return {"mode": mode, "seed": seed, "best_epoch": best["epoch"],
            "best_acc": max(h["val_accuracy"] for h in hist if h["epoch"] == best["epoch"]),
            "best_f1": best["f1"], "last_acc": hist[-1]["val_accuracy"], "last_f1": hist[-1]["val_macro_f1"],
            "last5_acc": float(np.mean([h["val_accuracy"] for h in last5])),
            "last5_f1": float(np.mean([h["val_macro_f1"] for h in last5])), "val_images": len(val_ds)}


if __name__ == "__main__":
    drop = conflict_set() if DROP_CONFLICT_PAIRS else set()
    if DROP_CONFLICT_PAIRS:
        print("Dropping", len(drop), "images that sit in two class folders")
    lab_ck = lab_pretrain() if "lab_pretrain" in MODES else None
    results = [train_field(m, s, drop, lab_ck) for m in MODES for s in SEEDS]
    df = pd.DataFrame(results)
    df.to_csv(OUT_RESULTS / "summary_by_run.csv", index=False)

    print("\n" + "=" * 70)
    print(f"SUMMARY (validation images: {df.val_images.iloc[0]}, conflicts dropped: {DROP_CONFLICT_PAIRS})")
    print("=" * 70)
    print(df.round(4).to_string(index=False))
    for m in MODES:
        d = df[df["mode"] == m]
        f = lambda c, pct=False: (f"{d[c].mean()*100:.1f}% +/- {d[c].std(ddof=0)*100:.1f}" if pct
                                  else f"{d[c].mean():.3f} +/- {d[c].std(ddof=0):.3f}")
        print(f"\n[{m}]")
        print("  Last-5 mean  accuracy:", f("last5_acc", True), "| macro F1:", f("last5_f1"), "  <- headline")
        print("  Last epoch   accuracy:", f("last_acc", True), "| macro F1:", f("last_f1"))
        print("  Best epoch   accuracy:", f("best_acc", True), "| macro F1:", f("best_f1"), "  (optimistic)")
    print("\nIf the two modes differ by less than ~3 points, treat them as equal.")