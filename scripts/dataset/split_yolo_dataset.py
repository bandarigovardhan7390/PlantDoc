"""
STEP 1 - Safe re-split of a YOLO-format dataset (FieldPlant / Roboflow export)

What it does
  1. Finds your dataset (PLANT/confirm), reads data.yaml for the class names.
     It pools every split folder it finds (train / valid / test), so the old split is ignored.
  2. Pairs every image with its label file. Images with NO label file are left out and listed.
     (An EMPTY label file is kept as a "no disease visible" background image.)
  3. Merges duplicate class names (CLASS_MERGE) and can drop classes (DROP_CLASSES).
  4. Finds copies of the same photo: exact copies, near-copies (resized / re-saved), and Roboflow
     augmentation copies (same name before ".rf."). Copies are kept together in ONE split, so the
     validation set can never contain a photo that is also in training.
  5. Splits by photo (not by leaf), balancing each class as well as possible, 80 / 10 / 10.
  6. Copies images + corrected labels into a NEW folder and writes a ready data.yaml.
  7. Writes audit files and prints a summary.

Your original dataset is never changed. The script refuses to run if the output folder exists.

Usage:  python split_yolo_dataset.py
        python split_yolo_dataset.py  <source_folder>  <output_folder>      (optional)
"""
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

# ------------------------- SETTINGS -------------------------
PROJECT_DIR = Path(r"D:\personal\project\PLANT")
SOURCE_CANDIDATES = [PROJECT_DIR / "confirm", PROJECT_DIR / "cofirm"]   # first one that exists
OUT_DIR = PROJECT_DIR / "fieldplant_split_v1"   # new folder, the first split is left untouched

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.80, 0.10, 0.10   # set TEST_RATIO = 0 and VAL_RATIO = 0.20 for train/valid only
SEED = 42
NEAR_DUP_BITS = 6          # 0-64: two photos closer than this are treated as copies (6 = fairly strict)
# old name -> name to use. Too few examples to learn mosaic and yellow virus on tomato separately,
# so they are merged into one "Tomato virus" class (remove these two lines to keep them separate).
CLASS_MERGE = {"Manioc_Mosaique": "Cassava Mosaic",
               "Tomato leaf mosaic virus": "Tomato virus", "Tomato leaf yellow virus": "Tomato virus"}
# classes with fewer than ~40 labelled leaves in the whole dataset (cannot be learned or tested)
DROP_CLASSES = ["Cassava Bacterial Blight", "Corn Charcoal", "Corn Chlorotic Leaf Spot", "Corn Insects Damages",
                "Corn Purple Discoloration", "Corn Smut", "Corn Violet Decoloration", "Corn Yellow Spots",
                "Tomato bacterial wilt"]
# a photo and its near-copy carrying DIFFERENT classes (38 pairs in your first audit): leaves labelled in one
# copy look like background in the other, which confuses training. True = leave those photos out.
EXCLUDE_CONFLICT_GROUPS = True
MIN_INSTANCES_WARN = 50    # warn when a class has fewer labelled leaves than this
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_FOLDERS = ["train", "valid", "val", "validation", "test"]
# ------------------------------------------------------------

if len(sys.argv) >= 3:
    SOURCE_CANDIDATES = [Path(sys.argv[1])]
    OUT_DIR = Path(sys.argv[2])

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def read_names(src):
    f = src / "data.yaml"
    if not f.exists():
        return None
    text = f.read_text(encoding="utf-8", errors="ignore")
    try:
        import yaml
        names = yaml.safe_load(text).get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        return [str(n) for n in names]
    except Exception:
        import re
        m = re.search(r"names\s*:\s*\[(.*?)\]", text, re.S)
        if m:
            return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
    return None


def collect(src):
    items = []
    for sp in SPLIT_FOLDERS:
        d = src / sp
        if not d.is_dir():
            continue
        img_dir = d / "images" if (d / "images").is_dir() else d
        lbl_dir = d / "labels" if (d / "labels").is_dir() else img_dir
        for p in sorted(img_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                lp = lbl_dir / (p.stem + ".txt")
                items.append({"img": p, "lbl": lp if lp.exists() else None, "orig_split": sp})
    return items


def parse_label(lp, n_old):
    """-> (list of (old_class, rest_of_line), list of problems)"""
    rows, probs = [], []
    for i, line in enumerate(lp.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        t = line.split()
        if not t:
            continue
        try:
            c = int(float(t[0]))
            vals = [float(x) for x in t[1:]]
        except ValueError:
            probs.append(f"line {i}: not numbers")
            continue
        if not (0 <= c < n_old):
            probs.append(f"line {i}: class id {c} out of range")
        elif len(vals) < 4 or any(v < -1e-6 or v > 1 + 1e-6 for v in vals):
            probs.append(f"line {i}: coordinates missing or outside 0-1")
        else:
            rows.append((c, " ".join(t[1:]), len(vals)))
    return rows, probs


def dhash(path):
    with Image.open(path) as im:
        im.draft("RGB", (160, 160))
        g = im.convert("L").resize((9, 8), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.frombuffer(np.packbits(bits).tobytes(), dtype=">u8")[0].astype(np.uint64)


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    src = next((c for c in SOURCE_CANDIDATES if c.is_dir()), None)
    if src is None:
        raise FileNotFoundError(f"Dataset folder not found. Tried: {SOURCE_CANDIDATES}")
    if OUT_DIR.exists():
        raise FileExistsError(f"{OUT_DIR} already exists. Rename or delete it first (nothing was changed).")
    print("Source:", src)

    names_old = read_names(src)
    items = collect(src)
    if not items:
        raise RuntimeError(f"No images found under {src} (looked for {SPLIT_FOLDERS}, with or without an images/ sub-folder)")
    print(f"Images found: {len(items)}  | class names in data.yaml: {len(names_old) if names_old else 'NOT FOUND'}")

    # ---- class remap
    if names_old is None:
        mx = 0
        for it in items:
            if it["lbl"]:
                for line in it["lbl"].read_text(errors="ignore").splitlines():
                    if line.split():
                        mx = max(mx, int(float(line.split()[0])))
        names_old = [f"class_{i}" for i in range(mx + 1)]
        print("WARNING: no data.yaml, using generic class names")
    merged = [CLASS_MERGE.get(n, n) for n in names_old]
    new_names = []
    for n in merged:
        if n not in DROP_CLASSES and n not in new_names:
            new_names.append(n)
    remap = {i: (new_names.index(n) if n in new_names else None) for i, n in enumerate(merged)}
    K = len(new_names)

    # ---- read labels
    problems, kept = [], []
    no_label, all_dropped, unreadable = [], [], []
    for it in items:
        if it["lbl"] is None:
            no_label.append(it["img"])
            continue
        rows, probs = parse_label(it["lbl"], len(names_old))
        problems += [f"{it['lbl']}: {p}" for p in probs]
        had = len(rows)
        out = [(remap[c], rest, nv) for c, rest, nv in rows if remap[c] is not None]
        if had > 0 and not out:
            all_dropped.append(it["img"])
            continue
        it["rows"] = out
        kept.append(it)
    print(f"Left out: {len(no_label)} without label file, {len(all_dropped)} whose only classes were dropped")

    # ---- hashes (exact + near) and groups
    print("Hashing images ...")
    good, md5s, h64 = [], [], []
    for n, it in enumerate(kept, 1):
        try:
            h = dhash(it["img"])
            m = hashlib.md5(it["img"].read_bytes()).hexdigest()
        except (UnidentifiedImageError, OSError, ValueError) as e:
            unreadable.append((it["img"], str(e)))
            continue
        good.append(it)
        md5s.append(m)
        h64.append(h)
        if n % 500 == 0:
            print(f"  {n}/{len(kept)}")
    kept = good
    N = len(kept)
    h64 = np.array(h64, dtype=np.uint64)
    dsu = DSU(N)
    by_md5, by_key = defaultdict(list), defaultdict(list)
    for i, it in enumerate(kept):
        by_md5[md5s[i]].append(i)
        key = it["img"].name.split(".rf.")[0] if ".rf." in it["img"].name else None
        if key:
            by_key[key].append(i)
    for d in (by_md5, by_key):
        for idx in d.values():
            for j in idx[1:]:
                dsu.union(idx[0], j)
    near = 0
    for i in range(N - 1):
        bits = POP[np.bitwise_xor(h64[i], h64[i + 1:]).view(np.uint8).reshape(-1, 8)].sum(axis=1)
        for j in np.nonzero(bits < NEAR_DUP_BITS)[0]:
            dsu.union(i, i + 1 + int(j))
            near += 1
    groups = defaultdict(list)
    for i in range(N):
        groups[dsu.find(i)].append(i)
    glist = list(groups.values())
    multi = [g for g in glist if len(g) > 1]
    print(f"Photos: {N} | groups: {len(glist)} | groups with copies: {len(multi)} covering {sum(len(g) for g in multi)} files")

    # ---- group x class matrix
    vec = np.zeros((len(glist), K), dtype=np.int64)
    n_img = np.array([len(g) for g in glist])
    conflicts = []
    for gi, g in enumerate(glist):
        sets = []
        for i in g:
            s = set()
            for c, _, _ in kept[i]["rows"]:
                vec[gi, c] += 1
                s.add(c)
            sets.append(frozenset(s))
        if len(set(sets)) > 1:
            conflicts.append((gi, g))

    excluded = {gi for gi, _ in conflicts} if EXCLUDE_CONFLICT_GROUPS else set()
    for gi in excluded:
        vec[gi, :] = 0
        n_img[gi] = 0
    N_eff = int(n_img.sum())
    if excluded:
        print(f"Leaving out {len(excluded)} groups ({sum(len(glist[g]) for g in excluded)} files) whose copies carry different classes")

    # ---- greedy balanced assignment
    ratios = {"train": TRAIN_RATIO, "valid": VAL_RATIO, "test": TEST_RATIO}
    ratios = {k: v for k, v in ratios.items() if v > 0}
    sp_names = list(ratios)
    r = np.array([ratios[s] for s in sp_names])
    r = r / r.sum()
    total_c = vec.sum(0).astype(float)
    target = np.outer(r, total_c)
    cur = np.zeros((len(sp_names), K))
    size = np.zeros(len(sp_names))
    rng = random.Random(SEED)
    order = [g for g in range(len(glist)) if g not in excluded]
    rng.shuffle(order)
    rarity = {g: (min((total_c[c] for c in range(K) if vec[g, c] > 0), default=1e12)) for g in order}
    order.sort(key=lambda g: rarity[g])
    assign = {}
    for g in order:
        v = vec[g]
        deficit = (target - cur) / np.maximum(target, 1.0)
        img_def = (r * N_eff - size) / (r * N_eff)
        score = (deficit * (v > 0)).sum(axis=1) + 0.2 * img_def
        s = int(np.argmax(score))
        assign[g] = s
        cur[s] += v
        size[s] += n_img[g]

    # ---- write output
    print("Copying files ...")
    for s in sp_names:
        (OUT_DIR / s / "images").mkdir(parents=True)
        (OUT_DIR / s / "labels").mkdir(parents=True)
    used = defaultdict(int)
    rows_out = []
    box_area = defaultdict(list)
    per = {s: np.zeros(K, dtype=int) for s in sp_names}
    per_img = {s: np.zeros(K, dtype=int) for s in sp_names}
    for gi, g in enumerate(glist):
        if gi in excluded:
            continue
        s = sp_names[assign[gi]]
        for i in g:
            it = kept[i]
            stem = it["img"].stem
            used[(s, stem)] += 1
            if used[(s, stem)] > 1:
                stem = f"{stem}__{used[(s, stem)]}"
            shutil.copy2(it["img"], OUT_DIR / s / "images" / (stem + it["img"].suffix.lower()))
            lines = [f"{c} {rest}" for c, rest, _ in it["rows"]]
            (OUT_DIR / s / "labels" / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            for c in {c for c, _, _ in it["rows"]}:
                per_img[s][c] += 1
            for c, rest, nv in it["rows"]:
                per[s][c] += 1
                if nv == 4:
                    w, h = [float(x) for x in rest.split()[2:4]]
                    box_area[c].append(w * h)
            rows_out.append([str(it["img"]), s, it["orig_split"], gi, len(g)])

    yaml_txt = (f"path: {OUT_DIR.as_posix()}\ntrain: train/images\nval: valid/images\n"
                + ("test: test/images\n" if "test" in sp_names else "")
                + f"nc: {K}\nnames: {json.dumps(new_names)}\n")
    (OUT_DIR / "data.yaml").write_text(yaml_txt, encoding="utf-8")

    # ---- audit files
    aud = OUT_DIR / "audit"
    aud.mkdir()
    with open(aud / "assignment.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source_image", "new_split", "original_split", "group_id", "group_size"])
        w.writerows(rows_out)
    with open(aud / "class_counts.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class"] + [f"{s}_leaves" for s in sp_names] + [f"{s}_images" for s in sp_names] + ["median_box_area_fraction"])
        for c in range(K):
            med = float(np.median(box_area[c])) if box_area[c] else ""
            w.writerow([new_names[c]] + [int(per[s][c]) for s in sp_names] + [int(per_img[s][c]) for s in sp_names] + [round(med, 4) if med != "" else ""])
    with open(aud / "copy_groups.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "split", "file"])
        for gi, g in enumerate(glist):
            if len(g) > 1 and gi not in excluded:
                for i in g:
                    w.writerow([gi, sp_names[assign[gi]], kept[i]["img"]])
    with open(aud / "class_conflicts.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "file", "classes_in_this_file"])
        for gi, g in conflicts:
            for i in g:
                w.writerow([gi, kept[i]["img"], " | ".join(sorted({new_names[c] for c, _, _ in kept[i]["rows"]}))])
    rep = ["LEFT OUT: no label file"] + [f"  {p}" for p in no_label]
    rep += ["LEFT OUT: all classes dropped"] + [f"  {p}" for p in all_dropped]
    rep += ["LEFT OUT: unreadable image"] + [f"  {p}: {e}" for p, e in unreadable]
    rep += ["LABEL PROBLEMS (bad lines were ignored)"] + [f"  {p}" for p in problems]
    (aud / "problems.txt").write_text("\n".join(rep), encoding="utf-8")

    # ---- summary
    print("\n" + "=" * 78)
    print("SPLIT SUMMARY  (leaves = labelled boxes, images = photos containing the class)")
    print("=" * 78)
    hdr = f"{'class':26s}" + "".join(f"{s[:5] + ' leaves':>13s}" for s in sp_names) + "".join(f"{s[:5] + ' imgs':>11s}" for s in sp_names)
    print(hdr)
    warn = []
    for c in range(K):
        print(f"{new_names[c]:26s}" + "".join(f"{per[s][c]:13d}" for s in sp_names) + "".join(f"{per_img[s][c]:11d}" for s in sp_names))
        if per[sp_names[0]][c] + sum(per[s][c] for s in sp_names[1:]) < MIN_INSTANCES_WARN:
            warn.append(f"{new_names[c]}: only {sum(per[s][c] for s in sp_names)} leaves in total")
        elif len(sp_names) > 1 and per[sp_names[1]][c] < 10:
            warn.append(f"{new_names[c]}: only {per[sp_names[1]][c]} leaves in validation (score will be very noisy)")
    print("-" * 78)
    print("Images per split:", {s: int(size[k]) for k, s in enumerate(sp_names)})
    print(f"Copies: {len(multi)} groups of copies (kept together) | groups whose files have DIFFERENT classes: {len(conflicts)}"
          + (" (left out)" if excluded else ""))
    if warn:
        print("\nWARNINGS:")
        for w_ in warn:
            print("  -", w_)
    print(f"\nLeft out: {len(no_label)} no label file | {len(all_dropped)} dropped classes | {len(unreadable)} unreadable | {len(problems)} bad label lines")
    print("\nNew dataset:", OUT_DIR)
    print("Use this file for training:", OUT_DIR / "data.yaml")
    print("Audit files:", aud)


if __name__ == "__main__":
    main()