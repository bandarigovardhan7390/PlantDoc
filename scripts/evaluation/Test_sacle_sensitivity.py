"""
Diagnostic: does the detector fail on CLOSE-UP photos (one leaf filling the whole frame)?

Your training photos show leaves that cover only about 7 to 13% of the picture. Close-ups are far bigger than that.
This script runs the SAME photo at several zoom levels: the photo is shrunk and centred on a grey canvas, so the leaf
looks smaller to the model. If the answer and confidence get better when the photo is shrunk, scale is the problem.

Put the photos you want to test in:   PLANT\\closeup_test   (and it also tests PLANT\\external_test)
Run:  python test_scale_sensitivity.py        (needs onnxruntime, and export_yolo26_onnx.py in the same folder)
"""
import csv
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_yolo26_onnx import decode, load_rgb, to_input   # noqa: E402

# ------------------------- SETTINGS -------------------------
PROJECT_DIR = Path(r"D:\personal\project\PLANT")
ONNX_FILE = PROJECT_DIR / "android_model" / "leaf_disease_yolo26n.onnx"
CLASS_FILE = PROJECT_DIR / "android_model" / "class_names.json"
IMAGE_DIRS = [PROJECT_DIR / "closeup_test", PROJECT_DIR / "external_test"]
OUT_CSV = PROJECT_DIR / "results" / "yolo26_disease" / "scale_sensitivity.csv"
SHRINKS = [1.0, 0.7, 0.5, 0.35, 0.25]    # 1.0 = what the app does today
CONF = 0.10                              # low on purpose, to see what the model almost chose
CROP_PREFIX = "Tomato"                   # same crop filter as the app
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE, PAD = 640, 114
# ------------------------------------------------------------


def shrunk_letterbox(img, shrink):
    w, h = img.size
    r = min(SIZE / w, SIZE / h) * shrink
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    canvas = Image.new("RGB", (SIZE, SIZE), (PAD, PAD, PAD))
    left, top = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (left, top))
    return canvas, r, left, top


def main():
    import onnxruntime as ort
    names = json.loads(CLASS_FILE.read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(ONNX_FILE), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    files = []
    for d in IMAGE_DIRS:
        if d.is_dir():
            files += sorted(p for p in d.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise RuntimeError(f"No photos found in {IMAGE_DIRS}")

    head = f"{'photo':34s} {'box%':>5s} " + " ".join(f"{'zoom ' + str(s):>22s}" for s in SHRINKS)
    print(head)
    print("-" * len(head))
    rows, better, worse, same = [], 0, 0, 0
    for p in files:
        img = load_rgb(p)
        cells, top_by_s = [], {}
        box_pct = ""
        for s in SHRINKS:
            canvas, r, left, top = shrunk_letterbox(img, s)
            dets = [d for d in decode(sess.run(None, {inp: to_input(canvas)})[0], CONF, r, left, top, *img.size)
                    if names[d[0]].startswith(CROP_PREFIX)]
            if dets:
                c, sc, b = dets[0]
                top_by_s[s] = (names[c].replace("Tomato ", ""), sc)
                cells.append(f"{top_by_s[s][0][:14]} {sc * 100:3.0f}%")
                if s == 1.0:
                    box_pct = f"{100 * (b[2] - b[0]) * (b[3] - b[1]) / (img.width * img.height):.0f}"
            else:
                top_by_s[s] = ("-", 0.0)
                cells.append("-")
        c1 = top_by_s[1.0][1]
        cshrunk = max(v[1] for k, v in top_by_s.items() if k != 1.0)   # best of the shrunk versions only
        if cshrunk > c1 + 0.10:
            better += 1
        elif cshrunk < c1 - 0.10:
            worse += 1
        else:
            same += 1
        print(f"{p.name[:34]:34s} {box_pct:>5s} " + " ".join(f"{c:>22s}" for c in cells))
        rows.append([p.name, box_pct] + cells)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["photo", "box_percent_of_photo"] + [f"zoom_{s}" for s in SHRINKS])
        w.writerows(rows)
    print("-" * len(head))
    print(f"Best shrunk version vs zoom 1.0 (top confidence): rises by 10+ points on {better} of {len(files)} photos | "
          f"drops by 10+: {worse} | about the same: {same}")
    print("box% = how much of the photo the detected leaf box covers at zoom 1.0 (training leaves cover about 7 to 13%).")
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()