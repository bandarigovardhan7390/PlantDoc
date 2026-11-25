"""
STEP 4a - Export the trained YOLO26 model for the Android app, and PROVE it matches the PC model

Your app already uses ONNX Runtime, so the disease detector goes into the app as an .onnx file.
(Llama will use ExecuTorch later - that is a separate pipeline.)

This script:
  1. Exports best.pt -> ONNX (fixed 640x640 input, end-to-end output [1, 300, 6] = x1,y1,x2,y2,confidence,class_id).
  2. Writes class_names.json and model_info.json (copy them into the app's assets folder).
  3. Runs the ONNX file with ONNX Runtime using EXACTLY the pre-processing the phone will use
     (fix photo rotation, letterbox to 640 with grey padding, divide by 255, NO ImageNet mean/std)
     and compares it with the PyTorch model on your external photos + some validation photos.
  4. Saves a few photos with the expected answers (parity_test/) so the phone can be checked the same way.

Install once:  pip install onnxruntime onnx
Run:           python export_yolo26_onnx.py
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# ------------------------- SETTINGS -------------------------
PROJECT_DIR = Path(r"D:\personal\project\PLANT")
WEIGHTS = PROJECT_DIR / "results" / "yolo26_disease" / "yolo26n_seed42" / "weights" / "best.pt"
OUT_DIR = PROJECT_DIR / "android_model"
EXTERNAL_DIR = PROJECT_DIR / "external_test"
VAL_IMAGES = PROJECT_DIR / "fieldplant_split_v1" / "valid" / "images"
N_VAL_IMAGES = 12
IMGSZ = 640
CONF = 0.25
PAD_VALUE = 114
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ------------------------------------------------------------


def load_rgb(path):
    """Open a photo and apply its EXIF rotation (the current app does NOT do this - phone photos can arrive sideways)."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def letterbox(img, size=IMGSZ, pad=PAD_VALUE):
    """Resize keeping the shape, centre on a size x size grey canvas. Returns canvas, scale, left, top."""
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    canvas = Image.new("RGB", (size, size), (pad, pad, pad))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (left, top))
    return canvas, r, left, top


def to_input(canvas):
    """RGB uint8 -> float32 [1,3,640,640], values 0..1 (YOLO needs NO mean/std)."""
    a = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])


def decode(out, conf, r, left, top, w, h):
    """[1,300,6] model output -> list of (class_id, confidence, [x1,y1,x2,y2]) in ORIGINAL photo pixels."""
    dets = []
    for row in out[0]:
        x1, y1, x2, y2, c, k = [float(v) for v in row]   # plain Python floats (JSON cannot store numpy float32)
        if c < conf:
            continue
        box = [(x1 - left) / r, (y1 - top) / r, (x2 - left) / r, (y2 - top) / r]
        box = [min(max(box[0], 0), w), min(max(box[1], 0), h), min(max(box[2], 0), w), min(max(box[3], 0), h)]
        dets.append((int(k), float(c), box))
    return sorted(dets, key=lambda d: -d[1])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    import onnxruntime as ort
    from ultralytics import YOLO

    if not WEIGHTS.exists():
        raise FileNotFoundError(WEIGHTS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    names = [model.names[i] for i in range(len(model.names))]

    # ---- 1. export
    exported = model.export(format="onnx", imgsz=IMGSZ, batch=1, dynamic=False, half=False, opset=17,
                            simplify=False, end2end=True, nms=False, device="cpu")
    onnx_path = OUT_DIR / "leaf_disease_yolo26n.onnx"
    shutil.copy2(exported, onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp, outp = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"ONNX file: {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Input : name='{inp.name}' shape={inp.shape} type={inp.type}")
    print(f"Output: name='{outp.name}' shape={outp.shape} type={outp.type}")
    if outp.shape[-1] != 6:
        print("WARNING: output is not [1,300,6]. Send me this printout before we write the Android code.")

    # ---- 2. files for the app
    (OUT_DIR / "class_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    (OUT_DIR / "model_info.json").write_text(json.dumps({
        "input_name": inp.name, "input_shape": [1, 3, IMGSZ, IMGSZ], "output_name": outp.name,
        "output_format": "[1,300,6] = x1,y1,x2,y2,confidence,class_id in 640x640 letterboxed pixels",
        "preprocess": "EXIF rotate, letterbox keep ratio centred, pad value 114, RGB, /255, NCHW, no mean/std",
        "confidence_threshold": CONF}, indent=2), encoding="utf-8")

    # ---- 3. parity check on PC
    imgs = sorted(p for p in EXTERNAL_DIR.rglob("*") if p.suffix.lower() in EXTS) if EXTERNAL_DIR.is_dir() else []
    if VAL_IMAGES.is_dir():
        v = sorted(p for p in VAL_IMAGES.iterdir() if p.suffix.lower() in EXTS)
        imgs += v[:: max(1, len(v) // N_VAL_IMAGES)][:N_VAL_IMAGES]
    if not imgs:
        raise RuntimeError("No images found for the parity check")
    parity_dir = OUT_DIR / "parity_test"
    parity_dir.mkdir(exist_ok=True)
    expected, times = {}, []
    same_top = n_cmp = 0
    diffs = []
    print(f"\n{'image':34s} {'ONNX top':26s} {'PyTorch top':26s} {'boxes':>7s}  match")
    print("-" * 104)
    for k, p in enumerate(imgs):
        img = load_rgb(p)
        canvas, r, left, top = letterbox(img)
        t0 = time.perf_counter()
        out = sess.run(None, {inp.name: to_input(canvas)})[0]
        times.append((time.perf_counter() - t0) * 1000)
        d_onnx = decode(out, CONF, r, left, top, *img.size)
        res = model.predict(img, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        d_pt = sorted([(int(c), float(s), xy) for c, s, xy in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(),
                                                                  res.boxes.xyxy.tolist())], key=lambda d: -d[1])
        top_o = f"{names[d_onnx[0][0]]} {d_onnx[0][1]*100:.0f}%" if d_onnx else "-"
        top_p = f"{names[d_pt[0][0]]} {d_pt[0][1]*100:.0f}%" if d_pt else "-"
        ok_top = (not d_onnx and not d_pt) or (d_onnx and d_pt and d_onnx[0][0] == d_pt[0][0])
        matched = sum(1 for c, s, b in d_onnx if any(c == c2 and iou(b, b2) > 0.5 for c2, s2, b2 in d_pt))
        same_top += bool(ok_top)
        n_cmp += 1
        if d_onnx and d_pt:
            diffs.append(abs(d_onnx[0][1] - d_pt[0][1]))
        print(f"{p.name[:34]:34s} {top_o:26s} {top_p:26s} {matched:3d}/{len(d_onnx):<3d}  {'OK' if ok_top else 'DIFFERENT'}")
        if k < 6:  # a few photos + expected answers for checking the phone later
            shutil.copy2(p, parity_dir / p.name)
            expected[p.name] = [{"class": names[c], "confidence": round(s, 4), "box_xyxy": [round(x, 1) for x in b]}
                                for c, s, b in d_onnx]
    (parity_dir / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    print("-" * 104)
    print(f"Same top class on {same_top} of {n_cmp} photos | mean difference in top confidence: "
          f"{(np.mean(diffs) * 100 if diffs else 0):.1f} points | ONNX Runtime on this PC: {np.mean(times[1:] or times):.0f} ms per photo")
    print("Small differences are normal (padding and resizing differ slightly). A DIFFERENT top class on many photos is not.")
    print("\nCopy these into the app's assets folder:", OUT_DIR / "leaf_disease_yolo26n.onnx", "and class_names.json")


if __name__ == "__main__":
    main()