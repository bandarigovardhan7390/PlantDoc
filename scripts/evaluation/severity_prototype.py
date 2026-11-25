"""
STEP 5a (version 2) - Severity prototype, no training: how much of each leaf looks diseased?

For every leaf box the colour of each pixel is sorted into
    green   = healthy leaf tissue
    yellow  = yellowing (chlorosis)                  -> affected
    brown   = brown / dead tissue                    -> affected
    purple  = purple / dark-red lesions              -> affected
    dark    = black spots, only if ENCLOSED by leaf  -> affected
    other   = background, shadow, glare              -> ignored
Version 2 fixes what version 1 got wrong on your photos:
  - brown straw / soil that is NOT joined to a green leaf is thrown away (only colour regions that contain green are the leaf)
  - black and purple lesions are now counted
  - leaves the detector calls "healthy" are not given a severity at all (a healthy leaf can never show "Severe")
severity = affected pixels / leaf pixels  ->  Mild / Moderate / Severe

It runs three checks:
  1. VALIDATION LABELS: uses the true leaf boxes of your validation images. Healthy leaves should score low and
     diseased leaves higher. The per-class table tells you whether the rule is sensible, with no hand-scoring.
  2. EXTERNAL PHOTOS: uses the ONNX detector on your external_test folder (tomato classes only).
  3. END TO END ON THE TEST SPLIT: the DETECTOR finds the leaves (not the true boxes), severity is measured only inside
     each detected box, and each detected box is matched to the true leaf so you can see severity by TRUE class.
Both save overlay pictures (affected pixels tinted red) so you can SEE what the rule counted.

This is a rough estimate. Brown soil, straw or a hand inside a box can be counted as disease.
The band limits are placeholders until you compare them with a few leaves scored by hand.

Install: pip install onnxruntime   (only needed for part 2)       Run: python severity_prototype.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ------------------------- SETTINGS -------------------------
PROJECT_DIR = Path(r"D:\personal\project\PLANT")
DATA_DIR = PROJECT_DIR / "fieldplant_split_v1"
VAL_IMAGES, VAL_LABELS = DATA_DIR / "valid" / "images", DATA_DIR / "valid" / "labels"
ONNX_FILE = PROJECT_DIR / "android_model" / "leaf_disease_yolo26n.onnx"
CLASS_FILE = PROJECT_DIR / "android_model" / "class_names.json"
EXTERNAL_DIR = PROJECT_DIR / "external_test"
OUT_DIR = PROJECT_DIR / "results" / "severity_prototype"

MILD_MAX, MODERATE_MAX = 0.10, 0.25    # <10% mild, 10-25% moderate, >25% severe  (placeholders)
MIN_LEAF_SHARE = 0.15                  # if fewer than 15% of the box looks like leaf, the number is "unreliable"
MAX_SIDE = 960                         # photos are shrunk to this longest side before measuring
N_VAL_OVERLAYS = 60
CONF, CROP_PREFIX = 0.25, "Tomato"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ------------------------------------------------------------


def rgb_to_hsv(a):
    """a: float32 [H,W,3] in 0..1 -> hue 0..360, saturation 0..1, value 0..1"""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    rc = (mx == r) & m
    gc = (mx == g) & m & ~rc
    bc = (mx == b) & m & ~rc & ~gc
    h[rc] = (60 * ((g - b)[rc] / d[rc])) % 360
    h[gc] = 60 * ((b - r)[gc] / d[gc]) + 120
    h[bc] = 60 * ((r - g)[bc] / d[bc]) + 240
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return h, s, mx


def leaf_masks(crop_u8):
    """uint8 RGB crop -> green, yellow, brown, purple, dark boolean masks"""
    h, s, v = rgb_to_hsv(crop_u8.astype(np.float32) / 255.0)
    green = (h >= 65) & (h <= 170) & (s >= 0.18) & (v >= 0.12)
    yellow = (h >= 40) & (h < 65) & (s >= 0.30) & (v >= 0.30)
    brown = ((h < 40) | (h >= 340)) & (s >= 0.20) & (v >= 0.12) & (v <= 0.80)
    purple = (h >= 260) & (h < 340) & (s >= 0.15) & (v >= 0.12)
    dark = v < 0.25
    return green, yellow, brown, purple, dark


def leaf_severity(crop_u8):
    g, y, b, p, dark = leaf_masks(crop_u8)
    area = crop_u8.shape[0] * crop_u8.shape[1]
    plant = g | y | b | p
    # leaf = colour regions that contain green (straw and soil with no green in them are dropped), holes filled
    joined = ndi.binary_closing(plant, structure=np.ones((3, 3)), iterations=2)
    labels, n = ndi.label(joined)
    keep = np.zeros_like(plant)
    for i in range(1, n + 1):
        comp = labels == i
        if comp.sum() >= 0.01 * area and (g & comp).sum() >= 0.05 * comp.sum():
            keep |= comp
    leaf = ndi.binary_fill_holes(keep)
    affected = ((y | b | p) & leaf) | (dark & leaf & ~plant)    # dark pixels count only when enclosed by the leaf
    leaf_px = int(leaf.sum())
    frac = float(affected.sum()) / leaf_px if leaf_px > 0 else 0.0
    return {"severity": frac, "leaf_share": leaf_px / area if area else 0.0, "affected_mask": affected,
            "reliable": area > 0 and leaf_px / area >= MIN_LEAF_SHARE}


def band(frac):
    return "Mild" if frac < MILD_MAX else ("Moderate" if frac < MODERATE_MAX else "Severe")


BAND_COLOR = {"Mild": (60, 180, 75), "Moderate": (255, 165, 0), "Severe": (230, 40, 40)}


def load_scaled(path):
    from export_yolo26_onnx import load_rgb
    img = load_rgb(path)
    s = min(1.0, MAX_SIDE / max(img.size))
    return img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR) if s < 1 else img


def measure(img, boxes):
    """boxes: list of (name, x1,y1,x2,y2) in pixels of img -> rows + annotated PIL image"""
    arr = np.asarray(img).copy()
    rows = []
    for idx, (name, x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(img.width, int(x2)), min(img.height, int(y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        r = leaf_severity(arr[y1:y2, x1:x2])
        m = r.pop("affected_mask")
        region = arr[y1:y2, x1:x2]
        region[m] = (0.55 * region[m] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
        rows.append({"class": name, "idx": idx, "box": (x1, y1, x2, y2), **r, "band": band(r["severity"])})
    out = Image.fromarray(arr)
    d = ImageDraw.Draw(out)
    for r in rows:
        x1, y1, x2, y2 = r["box"]
        col = BAND_COLOR[r["band"]] if r["reliable"] else (150, 150, 150)
        d.rectangle([x1, y1, x2, y2], outline=col, width=max(2, out.width // 300))
        d.text((x1 + 3, y1 + 3), f"{r['class']} {r['severity'] * 100:.0f}% {r['band'] if r['reliable'] else '(unreliable)'}", fill=col)
    return rows, out


def part1_validation(names):
    (OUT_DIR / "validation_overlays").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in VAL_IMAGES.iterdir() if p.suffix.lower() in EXTS)
    step = max(1, len(files) // N_VAL_OVERLAYS)
    allrows = []
    for k, p in enumerate(files):
        lp = VAL_LABELS / (p.stem + ".txt")
        if not lp.exists():
            continue
        img = load_scaled(p)
        boxes = []
        for line in lp.read_text().splitlines():
            t = line.split()
            if len(t) == 5:
                c, cx, cy, w, h = int(t[0]), *map(float, t[1:])
                boxes.append((names[c], (cx - w / 2) * img.width, (cy - h / 2) * img.height,
                              (cx + w / 2) * img.width, (cy + h / 2) * img.height))
        rows, out = measure(img, boxes)
        for r in rows:
            allrows.append({"image": p.name, **{k2: v for k2, v in r.items() if k2 not in ("box", "idx")}})
        if k % step == 0 and rows:
            out.save(OUT_DIR / "validation_overlays" / f"{p.stem[:60]}.jpg")
    with open(OUT_DIR / "validation_per_leaf.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "class", "severity", "leaf_share", "reliable", "band"])
        w.writeheader()
        w.writerows(allrows)

    print("\n" + "=" * 86)
    print("CHECK 1 - severity of leaves with TRUE boxes (validation split)")
    print("Healthy classes should be low; diseased classes higher.")
    print("=" * 86)
    print(f"{'class':26s} {'leaves':>6s} {'median':>8s} {'25%':>7s} {'75%':>7s} {'unreliable':>11s}")
    healthy, diseased = [], []
    for n in names:
        rr = [r for r in allrows if r["class"] == n]
        if not rr:
            continue
        ok = [r["severity"] * 100 for r in rr if r["reliable"]]
        if ok:
            print(f"{n:26s} {len(rr):6d} {np.median(ok):7.1f}% {np.percentile(ok, 25):6.1f}% {np.percentile(ok, 75):6.1f}% "
                  f"{100 * (1 - len(ok) / len(rr)):10.0f}%")
            (healthy if "healthy" in n.lower() else diseased).extend(ok)
    if healthy and diseased:
        print("-" * 86)
        print(f"Healthy leaves scored under 10%: {100 * np.mean(np.array(healthy) < 10):.0f}% | "
              f"diseased leaves scored 10% or more: {100 * np.mean(np.array(diseased) >= 10):.0f}%")


def part2_external(names):
    if not (ONNX_FILE.exists() and EXTERNAL_DIR.is_dir()):
        print("\n(Check 2 skipped: ONNX file or external_test folder not found)")
        return
    import onnxruntime as ort
    from export_yolo26_onnx import letterbox, to_input, decode
    sess = ort.InferenceSession(str(ONNX_FILE), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    (OUT_DIR / "external_overlays").mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 86)
    print("CHECK 2 - severity on EXTERNAL photos (leaves found by the detector, tomato classes only)")
    print("=" * 86)
    for p in sorted(q for q in EXTERNAL_DIR.rglob("*") if q.suffix.lower() in EXTS):
        img = load_scaled(p)
        canvas, r, left, top = letterbox(img)
        dets = [d for d in decode(sess.run(None, {inp: to_input(canvas)})[0], CONF, r, left, top, *img.size)
                if names[d[0]].startswith(CROP_PREFIX)]
        healthy_found = sum(1 for c, s, b in dets if "healthy" in names[c].lower())
        rows, out = measure(img, [(names[c], *b) for c, s, b in dets if "healthy" not in names[c].lower()])
        out.save(OUT_DIR / "external_overlays" / f"{p.stem[:60]}.jpg")
        if not rows and healthy_found:
            print(f"{p.name[:38]:38s} healthy leaves {healthy_found:2d} | no severity (healthy)")
        elif rows:
            worst = max(rows, key=lambda x: x["severity"])
            print(f"{p.name[:38]:38s} leaves {len(rows):2d} | worst: {worst['class']} {worst['severity'] * 100:.0f}% {worst['band']}"
                  f"{'' if worst['reliable'] else ' (unreliable)'}")
        else:
            print(f"{p.name[:38]:38s} no leaf found")


def part3_test_end_to_end(names):
    test_images, test_labels = DATA_DIR / "test" / "images", DATA_DIR / "test" / "labels"
    if not (ONNX_FILE.exists() and test_images.is_dir()):
        print("\n(Check 3 skipped: ONNX file or test split not found)")
        return
    import onnxruntime as ort
    from export_yolo26_onnx import letterbox, to_input, decode, iou
    sess = ort.InferenceSession(str(ONNX_FILE), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    (OUT_DIR / "test_overlays").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in test_images.iterdir() if p.suffix.lower() in EXTS)
    step = max(1, len(files) // N_VAL_OVERLAYS)
    allrows = []
    for k, p in enumerate(files):
        img = load_scaled(p)
        gt = []
        lp = test_labels / (p.stem + ".txt")
        if lp.exists():
            for line in lp.read_text().splitlines():
                t = line.split()
                if len(t) == 5:
                    c, cx, cy, w, h = int(t[0]), *map(float, t[1:])
                    gt.append((names[c], [(cx - w / 2) * img.width, (cy - h / 2) * img.height,
                                          (cx + w / 2) * img.width, (cy + h / 2) * img.height]))
        canvas, r, left, top = letterbox(img)
        dets = decode(sess.run(None, {inp: to_input(canvas)})[0], CONF, r, left, top, *img.size)
        scored = [(c, sc, b) for c, sc, b in dets if "healthy" not in names[c].lower()]
        rows, out = measure(img, [(names[c], *b) for c, sc, b in scored])
        for row in rows:
            c, sc, b = scored[row["idx"]]
            m = match_gt(b, gt, iou)
            allrows.append({"image": p.name, "predicted": names[c], "confidence": round(sc, 3), "true_class": m,
                            "severity": row["severity"], "reliable": row["reliable"], "band": row["band"]})
        if k % step == 0 and rows:
            out.save(OUT_DIR / "test_overlays" / f"{p.stem[:60]}.jpg")
    with open(OUT_DIR / "test_end_to_end_per_leaf.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "predicted", "confidence", "true_class", "severity", "reliable", "band"])
        w.writeheader()
        w.writerows(allrows)
    print("\n" + "=" * 86)
    print("CHECK 3 - end to end on the TEST split: severity inside DETECTED boxes, grouped by the TRUE class of the leaf")
    print("(leaves the detector calls healthy are not scored; 'no true leaf' = detector box that matches no labelled leaf)")
    print("=" * 86)
    print(f"{'true class':26s} {'leaves':>6s} {'median':>8s} {'25%':>7s} {'75%':>7s} {'severe':>8s}")
    for tc in names + ["no true leaf"]:
        rr = [x for x in allrows if x["true_class"] == tc and x["reliable"]]
        if rr:
            v = np.array([x["severity"] * 100 for x in rr])
            print(f"{tc:26s} {len(rr):6d} {np.median(v):7.1f}% {np.percentile(v, 25):6.1f}% {np.percentile(v, 75):6.1f}% "
                  f"{100 * np.mean(v >= MODERATE_MAX * 100):7.0f}%")
    hh = [x for x in allrows if x["true_class"] and "healthy" in x["true_class"].lower() and x["reliable"]]
    print("-" * 86)
    print(f"True-HEALTHY leaves the detector called diseased: {len(hh)}"
          + (f" | of those, scored Severe: {100 * np.mean([x['severity'] >= MODERATE_MAX for x in hh]):.0f}%" if hh else ""))


def match_gt(box, gt, iou_fn, min_iou=0.5):
    """True class of the labelled leaf that overlaps a detected box the most (IoU >= 0.5), else 'no true leaf'."""
    best, name = 0.0, "no true leaf"
    for n, g in gt:
        v = iou_fn(box, g)
        if v > best:
            best, name = v, n
    return name if best >= min_iou else "no true leaf"


def main():
    names = json.loads(next(l for l in (DATA_DIR / "data.yaml").read_text().splitlines() if l.startswith("names:")).split(":", 1)[1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    part1_validation(names)
    part2_external(names)
    part3_test_end_to_end(names)
    print("\nOverlays (red = counted as affected):", OUT_DIR)


if __name__ == "__main__":
    main()