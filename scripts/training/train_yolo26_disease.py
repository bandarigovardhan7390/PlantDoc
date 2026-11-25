"""
STEP 2 - Train the YOLO26 disease detector (baseline) on fieldplant_split_v1

Trains one YOLO26 model (one leaf = one box + disease class), then scores the BEST checkpoint
on the validation split and on the TEST split. The test split was never used to pick anything,
so the test numbers are the honest ones.

First run with SMOKE_TEST = False (a few minutes): it only checks that data, GPU and scoring work.
Then set SMOKE_TEST = False for the real run.

Install once:  pip install -U ultralytics
Run:           python train_yolo26_disease.py

Outputs (results/yolo26_disease/<run>/): weights/best.pt, training plots, per_class_val.csv,
per_class_test.csv, summary.json
"""
import csv
import json
from pathlib import Path

import torch
from ultralytics import YOLO

# ------------------------- SETTINGS -------------------------
PROJECT_DIR = Path(r"D:\personal\project\PLANT")
DATA_DIR = PROJECT_DIR / "fieldplant_split_v1"
DATA_YAML = DATA_DIR / "data.yaml"
RESULTS_DIR = PROJECT_DIR / "results" / "yolo26_disease"

MODEL = "yolo26n.pt"      # n = smallest/fastest (best for the phone). Try "yolo26s.pt" later.
IMGSZ = 640               # leaves are small in the photos, so do not go below 640
EPOCHS = 100
BATCH = 16                # lower to 8 if you get an out-of-memory error
PATIENCE = 30             # stop early if validation does not improve for 30 epochs
WORKERS = 4               # set to 0 if Windows shows multiprocessing errors
SEEDS = [42]              # one seed for the first baseline; use [42, 1, 7] for the final comparison
SMOKE_TEST = False         # True = 3 epochs on 10% of the data, only to check everything runs
# ------------------------------------------------------------


def count_leaves(split_folder, names):
    counts = {n: 0 for n in names}
    for f in (DATA_DIR / split_folder / "labels").glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.split():
                counts[names[int(float(line.split()[0]))]] += 1
    return counts


def per_class_rows(metrics, names, leaves):
    b = metrics.box
    rows = []
    for k, c in enumerate(list(b.ap_class_index)):
        n = names[int(c)]
        rows.append({"class": n, "leaves": leaves.get(n, 0), "precision": round(float(b.p[k]), 3),
                     "recall": round(float(b.r[k]), 3), "map50": round(float(b.ap50[k]), 3),
                     "map50_95": round(float(b.ap[k]), 3)})
    return rows


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"{DATA_YAML} not found - run split_yolo_dataset.py first")
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name(0) if device == 0 else "CPU (training will be very slow)")
    for seed in SEEDS:
        run = f"{Path(MODEL).stem}_seed{seed}" + ("_smoke" if SMOKE_TEST else "")
        model = YOLO(MODEL)
        model.train(data=str(DATA_YAML), epochs=3 if SMOKE_TEST else EPOCHS, imgsz=IMGSZ, batch=BATCH,
                    seed=seed, patience=PATIENCE, workers=WORKERS, device=device,
                    fraction=0.1 if SMOKE_TEST else 1.0, project=str(RESULTS_DIR), name=run)
        run_dir = Path(model.trainer.save_dir)
        best = run_dir / "weights" / "best.pt"
        names = YOLO(str(best)).names
        names = [names[i] for i in range(len(names))]

        summary = {"run": run, "model": MODEL, "imgsz": IMGSZ, "seed": seed, "smoke_test": SMOKE_TEST}
        tables = {}
        for split, folder in (("val", "valid"), ("test", "test")):
            m = YOLO(str(best)).val(data=str(DATA_YAML), split=split, imgsz=IMGSZ, batch=BATCH,
                                    device=device, project=str(RESULTS_DIR), name=f"{run}_{split}", plots=True)
            rows = per_class_rows(m, names, count_leaves(folder, names))
            tables[split] = rows
            with open(run_dir / f"per_class_{split}.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            summary[split] = {"map50": round(float(m.box.map50), 4), "map50_95": round(float(m.box.map), 4),
                              "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4)}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("\n" + "=" * 86)
        print(f"RESULTS  {run}   (mAP50 = how well boxes and classes match; recall = share of leaves found)")
        print("=" * 86)
        print(f"{'class':26s} {'val leaves':>10s} {'val rec':>8s} {'val mAP50':>10s} | {'test leaves':>11s} {'test rec':>9s} {'test mAP50':>11s}")
        tt = {r["class"]: r for r in tables["test"]}
        for r in tables["val"]:
            t = tt.get(r["class"], {"leaves": 0, "recall": 0, "map50": 0})
            print(f"{r['class']:26s} {r['leaves']:10d} {r['recall']:8.3f} {r['map50']:10.3f} | "
                  f"{t['leaves']:11d} {t['recall']:9.3f} {t['map50']:11.3f}")
        print("-" * 86)
        for s in ("val", "test"):
            x = summary[s]
            print(f"ALL ({s:4s})  mAP50 {x['map50']:.3f} | mAP50-95 {x['map50_95']:.3f} | precision {x['precision']:.3f} | recall {x['recall']:.3f}")
        print("\nSaved to:", run_dir)
        if SMOKE_TEST:
            print("\nSMOKE TEST ONLY - numbers are meaningless. Set SMOKE_TEST = False for the real run.")


if __name__ == "__main__":
    main()