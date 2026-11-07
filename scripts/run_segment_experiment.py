"""
Step 2 (second downstream task): part/table segmentation, ideal vs realistic depth.

Usage:
    python scripts/run_segment_experiment.py               # tuning, then evaluation
    python scripts/run_segment_experiment.py --tune-only   # stop after tuning
    python scripts/run_segment_experiment.py --skip-tuning # reuse outputs/segmentation/chosen.json
    python scripts/run_segment_experiment.py --rechoose --tune-only  # choose again from the tuning CSVs

Same design as the pose experiment:
1. Tuning on the tuning scenes (--tune): every threshold t in T_GRID_MM, once on ideal depth
   ("tuned_on_ideal") and once on realistic depth ("tuned_on_realistic").

2. Evaluation on the evaluation scenes (--eval): every Step 1 camera variant with both t.
       A = tuned_on_ideal on ideal
       B = tuned_on_ideal on realistic
       C = tuned_on_realistic on realistic

A part is "found" if at least 50 % of its pixels are marked "part" (75 % reported as well).

Writes in --out: tuning_parts.csv, tuning_scenes.csv, chosen.json, evaluation_parts.csv, evaluation_scenes.csv.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # before Open3D is loaded (repeatable RANSAC)

import argparse
import csv
import json
import time
from multiprocessing import Pool

import numpy as np

from rgbd_sim.camera_model import VARIANTS
from rgbd_sim.scene import load_depth, load_scene
from rgbd_sim.segment import evaluate, segment_parts, true_table_plane

T_GRID_MM = [1, 2, 3, 4, 5, 6, 8, 12]
TUNED_ON = {"tuned_on_ideal": "ideal", "tuned_on_realistic": "realistic"}
PART_FIELDS = ["phase", "params", "variant", "scene", "t_mm", "instance_id", "cls",
               "visible_pixels", "recall", "found50", "found75"]
SCENE_FIELDS = ["phase", "params", "variant", "scene", "t_mm", "part_pixels_marked",
                "correct_pixels_marked", "precision", "plane_angle_deg", "plane_offset_mm", "seconds"]


def run_job(job):
    """Segment one scene in one camera variant with one t; per-part and per-scene results."""
    phase, params, variant, folder, t_mm = job
    t0 = time.perf_counter()
    scene = load_scene(folder)
    meta, label = scene["meta"], scene["label"]
    depth = load_depth(folder, variant)
    mask, (n, d) = segment_parts(depth, np.array(meta["camera"]["K"]), t_mm / 1000, seed=0)
    res = evaluate(mask, label, meta)
    n_true, d_true = true_table_plane(meta)
    name = os.path.basename(folder)
    visible = {o["instance_id"]: o["visible_pixels"] for o in meta["objects"]}
    parts = [dict(phase=phase, params=params, variant=variant, scene=name, t_mm=t_mm, instance_id=i,
                  cls=c, visible_pixels=visible[i], recall=r, found50=r >= 0.5, found75=r >= 0.75)
             for i, c, r, _ in res["parts"]]
    marked = int(mask.sum())
    correct = int((mask & (label >= 2)).sum())
    scene_row = dict(phase=phase, params=params, variant=variant, scene=name, t_mm=t_mm,
                     part_pixels_marked=marked, correct_pixels_marked=correct,
                     precision=correct / marked if marked else 0.0,
                     plane_angle_deg=float(np.degrees(np.arccos(np.clip(n @ n_true, -1, 1)))),
                     plane_offset_mm=1000 * (d - d_true), seconds=time.perf_counter() - t0)
    return parts, scene_row


def scene_folders(dataset):
    root = os.path.join(dataset, "scenes")
    return sorted(os.path.join(root, s) for s in os.listdir(root))


def run_all(jobs, workers, out, prefix):
    parts, scenes, t0 = [], [], time.time()
    with Pool(workers) as pool:
        for k, (p, s) in enumerate(pool.imap_unordered(run_job, jobs, chunksize=2), 1):
            parts += p
            scenes.append(s)
            if k % 50 == 0 or k == len(jobs):
                print(f"  {prefix}: {k}/{len(jobs)} jobs, {(time.time() - t0) / 60:.1f} min", flush=True)
    for name, rows, fields in ((f"{prefix}_parts.csv", parts, PART_FIELDS),
                               (f"{prefix}_scenes.csv", scenes, SCENE_FIELDS)):
        with open(os.path.join(out, name), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return parts, scenes


def pooled(parts, scenes):
    """Share of parts found (50 % and 75 %), mean part recall, pixel precision, F1 (pooled)."""
    found = float(np.mean([p["found50"] for p in parts])) if parts else float("nan")
    marked = sum(s["part_pixels_marked"] for s in scenes)
    precision = sum(s["correct_pixels_marked"] for s in scenes) / marked if marked else 0.0
    f1 = 0.0 if found + precision == 0 else 2 * found * precision / (found + precision)
    return {"n_parts": len(parts), "found50": found,
            "found75": float(np.mean([p["found75"] for p in parts])) if parts else float("nan"),
            "mean_recall": float(np.mean([p["recall"] for p in parts])) if parts else float("nan"),
            "precision": precision, "f1": f1}


def tune(args):
    jobs = [("tuning", params, variant, folder, t)
            for params, variant in TUNED_ON.items() for folder in scene_folders(args.tune) for t in T_GRID_MM]
    print(f"Tuning: {len(jobs)} jobs on {args.tune}")
    return run_all(jobs, args.workers, args.out, "tuning")


def read_csv(path, ints=(), floats=(), bools=()):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ints:
            r[k] = int(float(r[k]))
        for k in floats:
            r[k] = float(r[k])
        for k in bools:
            r[k] = r[k] == "True"
    return rows


def choose(parts, scenes, out):
    chosen = {}
    for params in TUNED_ON:
        print(f"\n{params}:   t [mm]   found (50 %)   precision   F1")
        results = {}
        for t in T_GRID_MM:
            s = pooled([p for p in parts if p["params"] == params and p["t_mm"] == t],
                       [x for x in scenes if x["params"] == params and x["t_mm"] == t])
            results[t] = s
            print(f"            {t:5d}   {s['found50']:12.3f}   {s['precision']:9.3f}   {s['f1']:.3f}")
        top = max(s["f1"] for s in results.values())
        ties = [t for t, s in results.items() if s["f1"] >= top - 1e-9]
        t = ties[(len(ties) - 1) // 2]                      # middle of the tied values
        chosen[params] = {"t_mm": t, "ties_mm": ties, **results[t]}
        edge = t in (T_GRID_MM[0], T_GRID_MM[-1])
        print(f"  chosen: t = {t} mm" + ("   (on the edge of the grid)" if edge else "")
              + (f"   tied with: {[x for x in ties if x != t]} mm (evaluated too)" if len(ties) > 1 else ""))
    with open(os.path.join(out, "chosen.json"), "w") as f:
        json.dump(chosen, f, indent=2)
    return chosen


def parameter_sets(chosen):
    """Name -> t: the chosen t of each tuning, plus every other tied value."""
    sets = {}
    for params, c in chosen.items():
        sets[params] = c["t_mm"]
        for t in c.get("ties_mm", []):
            if t != c["t_mm"]:
                sets[f"{params}_tie_t{t}"] = t
    return sets


def evaluate_all(args, chosen):
    sets = parameter_sets(chosen)
    jobs = [("evaluation", params, variant, folder, t)
            for folder in scene_folders(args.eval) for variant in VARIANTS for params, t in sets.items()]
    print(f"\nEvaluation: {len(jobs)} jobs on {args.eval}")
    parts, scenes = run_all(jobs, args.workers, args.out, "evaluation")
    print(f"\n{'params (t)':30s} {'variant':10s} {'class':8s} {'n':>4s} {'found50':>8s} {'found75':>8s} "
          f"{'recall':>7s} {'precision':>9s}")
    for params, t in sets.items():
        for variant in VARIANTS:
            sc = [x for x in scenes if x["params"] == params and x["variant"] == variant]
            for cls in ("all", "bracket", "profile"):
                pt = [p for p in parts if p["params"] == params and p["variant"] == variant
                      and (cls == "all" or p["cls"] == cls)]
                s = pooled(pt, sc)
                prec = f"{s['precision']:9.3f}" if cls == "all" else f"{'':9s}"   # pixels: all parts
                print(f"{params + f' ({t} mm)':30s} {variant:10s} {cls:8s} {s['n_parts']:4d} {s['found50']:8.3f} "
                      f"{s['found75']:8.3f} {s['mean_recall']:7.3f} {prec}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tune", default="outputs/dataset")
    parser.add_argument("--eval", default="outputs/dataset_eval")
    parser.add_argument("--out", default="outputs/segmentation")
    parser.add_argument("--workers", type=int, default=2,
                        help="parallel processes (default 2: leaves the CPU to other runs)")
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--rechoose", action="store_true", help="choose again from the tuning CSVs")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.skip_tuning:
        with open(os.path.join(args.out, "chosen.json")) as f:
            chosen = json.load(f)
    else:
        if args.rechoose:
            parts = read_csv(os.path.join(args.out, "tuning_parts.csv"), ints=("t_mm", "instance_id"),
                             floats=("recall",), bools=("found50", "found75"))
            scenes = read_csv(os.path.join(args.out, "tuning_scenes.csv"), ints=("t_mm",),
                              floats=("part_pixels_marked", "correct_pixels_marked"))
        else:
            parts, scenes = tune(args)
        chosen = choose(parts, scenes, args.out)
    if not args.tune_only:
        evaluate_all(args, chosen)
