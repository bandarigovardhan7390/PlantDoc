"""
STEP 3 - Test the trained YOLO26 detector on your EXTERNAL photos

Put the external photos in their own folder (EXTERNAL_DIR below), NOT inside fieldplant_split_v1.
For every photo it saves:
  - an annotated copy with a box, class name and confidence on every leaf found
  - one row in external_results.csv (leaves found per class, top detection, weaker candidates,
    and a match / no match against EXPECTED)
Nothing is trained or changed.

Run:  python test_yolo26_external.py
"""
import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

# ------------------------- SETTINGS -------------------------
PROJECT_DIR = Path(r"D:\personal\project\PLANT")
WEIGHTS = PROJECT_DIR / "results" / "yolo26_disease" / "yolo26n_seed42" / "weights" / "best.pt"
EXTERNAL_DIR = PROJECT_DIR / "external_test"
OUT_DIR = PROJECT_DIR / "results" / "yolo26_disease" / "external"
IMGSZ = 640
CONF = 0.25          # detections at or above this are "found"
LOW_CONF = 0.05      # weaker candidates are listed separately, to see what the model almost chose
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# TENTATIVE: your friend's labels turned into FieldPlant class names. Check and edit.
# FieldPlant's "Tomato blight leaf" and "Tomato Brown Spots" are broad names, so I assumed
#   late/early blight -> blight leaf | Septoria / bacterial spot -> Brown Spots | mosaic / TYLCV -> virus.
# A list means "any of these counts as right". A name ending in * matches the start of the file name.
EXPECTED = {
    "1": ["Tomato blight leaf"],
    "2": ["Tomato healthy"],
    "3": ["Tomato Brown Spots"],
    "3-s2.0*": ["Tomato Brown Spots"],
    "4": ["Tomato blight leaf"],
    "5": ["Tomato virus"],
    "6": ["Tomato virus"],
    "8": ["Tomato Brown Spots", "Tomato blight leaf"],      # mixed early blight + Septoria
    "Tomato leaf mosaic virus (3)": ["Tomato virus"],
}
SKIP = {"7"}   # label doubtful, not scored
# ------------------------------------------------------------


def expected_for(stem):
    for k, v in EXPECTED.items():
        if stem == k or (k.endswith("*") and stem.startswith(k[:-1])):
            return v
    return None


def detections(r, names):
    out = []
    for c, p, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        out.append((names[int(c)], float(p), [round(v) for v in xy]))
    return sorted(out, key=lambda d: -d[1])


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")
    if not EXTERNAL_DIR.is_dir():
        raise FileNotFoundError(f"Create {EXTERNAL_DIR} and copy your external photos into it")
    files = sorted(p for p in EXTERNAL_DIR.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    if not files:
        raise RuntimeError(f"No images in {EXTERNAL_DIR}")
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(WEIGHTS))
    names = {int(k): v for k, v in model.names.items()}
    (OUT_DIR / "annotated").mkdir(parents=True, exist_ok=True)

    rows, scored, correct = [], 0, 0
    print(f"{'image':34s} {'leaves':>6s}  {'top detection':34s} {'expected':26s} result")
    print("-" * 112)
    for f in files:
        r = model.predict(str(f), imgsz=IMGSZ, conf=CONF, device=device, verbose=False)[0]
        low = model.predict(str(f), imgsz=IMGSZ, conf=LOW_CONF, device=device, verbose=False)[0]
        found = detections(r, names)
        weak = [d for d in detections(low, names) if d[1] < CONF]
        Image.fromarray(r.plot()[..., ::-1]).save(OUT_DIR / "annotated" / f"{f.stem}.jpg")

        top = found[0] if found else None
        counts = Counter(d[0] for d in found)
        exp = expected_for(f.stem)
        if f.stem in SKIP:
            result = "not scored"
        elif exp is None:
            result = "no label"
        elif top is None:
            result = "NO DETECTION"
            scored += 1
        else:
            ok = top[0] in exp
            result = "match" if ok else "NO MATCH"
            scored += 1
            correct += ok
        top_txt = f"{top[0]} {top[1]*100:.0f}%" if top else "-"
        print(f"{f.name[:34]:34s} {len(found):6d}  {top_txt:34s} {' / '.join(exp)[:26] if exp else '-':26s} {result}")
        rows.append({"image": f.name, "leaves_found": len(found),
                     "top_class": top[0] if top else "", "top_confidence": round(top[1] * 100, 1) if top else "",
                     "leaves_per_class": " | ".join(f"{k}:{v}" for k, v in counts.most_common()),
                     "all_detections": " | ".join(f"{c}:{p*100:.0f}%" for c, p, _ in found),
                     "weaker_candidates_5_to_25pct": " | ".join(f"{c}:{p*100:.0f}%" for c, p, _ in weak[:6]),
                     "expected": " / ".join(exp) if exp else "", "result": result})
    with open(OUT_DIR / "external_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("-" * 112)
    print(f"Matched {correct} of {scored} scored images (top detection only; the mapping of labels is tentative).")
    print("Annotated images + CSV:", OUT_DIR)


if __name__ == "__main__":
    main()