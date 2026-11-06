"""The experiment: does the camera model change the outcome of pose estimation?

Usage (after generating both datasets and their camera variants, see M5):
    python scripts/run_pose_experiment.py                   # tuning, then evaluation
    python scripts/run_pose_experiment.py --tune-only       # stop after tuning (look at it first)
    python scripts/run_pose_experiment.py --skip-tuning     # reuse outputs/experiment/chosen.json
    python scripts/run_pose_experiment.py --rechoose --tune-only   # choose again from tuning.csv
    python scripts/run_pose_experiment.py --extend --tune-only     # run only the grid points
                                                     # missing from tuning.csv, then choose
Phases:
1. Tuning, on the tuning scenes only (--tune). For each part class, every combination of voxel
   size v and tolerance d in the grids below is run on every part, twice:
       "tuned_on_ideal" on the ideal depth, "tuned_on_realistic" on the realistic depth
       (all effects of the camera model).
   Kept: the combination with the highest ADD-S AUC (ties: lower median centre error).
   AUC = area under the curve "share of parts with ADD-S < t" for t from 0 to 10 % of the
   diameter, divided by that range = mean over parts of max(0, 1 - ADD-S / (0.1 diameter)).
   It rewards both success and accuracy (PoseCNN reports the same kind of AUC); the success
   rate alone saturated on the tuning scenes (>= 0.93 for every combination).
2. Evaluation, on the evaluation scenes only (--eval): every part, every camera variant, with
   both chosen parameter sets. The conditions of the report are rows of this table:
       A = tuned_on_ideal on ideal      B = tuned_on_ideal on realistic (and on each single effect)
       C = tuned_on_realistic on realistic
3. Summary per parameter set, variant and class.
Writes in --out: tuning.csv, chosen.json, evaluation.csv, summary.json.
Runs --workers processes in parallel, each with ONE thread (Open3D's RANSAC is only repeatable
single-threaded).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # must be set before Open3D is loaded

import argparse
import csv
import json
import time
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

from rgbd_sim.camera_model import VARIANTS
from rgbd_sim.metrics import (add_s, centre_error, diameter, metric_points, rotation_error,
                              success)
from rgbd_sim.pose import estimate_pose_verified, model_point_cloud
from rgbd_sim.scene import MAKE_PART, load_depth, load_scene

# Search grids (mm). Ranges chosen from pilot runs on tuning scenes 0-9 (see the report).
VOXEL_GRID_MM = {"bracket": [2, 3, 4, 5, 6], "profile": [6, 8, 10, 12]}
# d for brackets extended to 12, 16 mm: with [1, 2, 4, 8] the choice on the realistic depth fell
# on d = 8 with the AUC still rising (0.797, 0.806, 0.814, 0.822 at v = 4 mm).
TOLERANCE_GRID_MM = {"bracket": [1, 2, 4, 8, 12, 16], "profile": [1, 2, 4, 8]}
TUNED_ON = {"tuned_on_ideal": "ideal", "tuned_on_realistic": "realistic"}
MIN_POINTS = 10          # fewer measured points than this: no estimate (counted as failure)

FIELDS = ["phase", "params", "variant", "scene", "instance_id", "cls", "visible_pixels",
          "points", "voxel_mm", "tolerance_mm", "add_s_mm", "success", "centre_mm",
          "rotation_deg", "score", "seconds", "error"]

# ---------- work done inside each process ----------
_parts = {}


def _init_worker():
    """Load the part models once per process."""
    for cls, make in MAKE_PART.items():
        mesh = make()
        _parts[cls] = {"mesh": mesh, "model": model_point_cloud(mesh),
                       "points": metric_points(mesh), "diameter": diameter(mesh)}


@lru_cache(maxsize=4)
def _scene(folder):
    return load_scene(folder)


@lru_cache(maxsize=8)
def _depth(folder, variant):
    return load_depth(folder, variant)


def run_job(job):
    """Estimate the pose of one part in one camera variant with one (v, d); measure the error."""
    phase, params, variant, folder, inst, cls, v_mm, d_mm = job
    scene, depth = _scene(folder), _depth(folder, variant)
    meta = scene["meta"]
    obj = next(o for o in meta["objects"] if o["instance_id"] == inst)
    n_points = int(((scene["label"] == inst) & (depth > 0)).sum())
    row = dict(phase=phase, params=params, variant=variant, scene=os.path.basename(folder),
               instance_id=inst, cls=cls, visible_pixels=obj["visible_pixels"], points=n_points,
               voxel_mm=v_mm, tolerance_mm=d_mm, add_s_mm=np.nan, success=False,
               centre_mm=np.nan, rotation_deg=np.nan, score=np.nan, seconds=0.0, error="")
    if n_points < MIN_POINTS:
        row["error"] = "too few points"
        return row
    part = _parts[cls]
    true = np.array(obj["pose_camera"])
    try:
        pose, info = estimate_pose_verified(depth, scene["label"], inst, np.array(meta["camera"]["K"]),
                                            part["mesh"], part["model"], v_mm / 1000,
                                            tolerance=d_mm / 1000, seed=0)
    except Exception as e:                       # one failing part must not stop the run
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    a = add_s(pose, true, part["mesh"], part["points"])
    row.update(add_s_mm=1000 * a, success=success(a, part["diameter"]),
               centre_mm=1000 * centre_error(pose, true),
               rotation_deg=rotation_error(pose, true, cls),
               score=info["score"], seconds=info["seconds"])
    return row


# ---------- helpers for the main process ----------
def scene_folders(dataset):
    root = os.path.join(dataset, "scenes")
    return sorted(os.path.join(root, d) for d in os.listdir(root))


def parts_of(folder):
    with open(os.path.join(folder, "meta.json")) as f:
        return [(o["instance_id"], o["class"]) for o in json.load(f)["objects"]]


def run_all(jobs, workers, path, label, append=False):
    """Run jobs in parallel, write every result row to a CSV file, return the rows."""
    rows, t0 = [], time.time()
    with Pool(workers, initializer=_init_worker) as pool, open(path, "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not append:
            writer.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_job, jobs, chunksize=4), 1):
            writer.writerow(row)
            rows.append(row)
            if i % 100 == 0 or i == len(jobs):
                elapsed = time.time() - t0
                print(f"  {label}: {i}/{len(jobs)} jobs, {elapsed / 60:.1f} min, "
                      f"about {elapsed / i * (len(jobs) - i) / 60:.1f} min left", flush=True)
    return rows


DIAMETER_MM = {}         # filled in the main process


def auc(rows):
    """ADD-S AUC up to 10 % of the diameter: mean of max(0, 1 - ADD-S / threshold); no estimate = 0."""
    vals = [0.0 if np.isnan(r["add_s_mm"]) else
            max(0.0, 1.0 - r["add_s_mm"] / (0.1 * DIAMETER_MM[r["cls"]])) for r in rows]
    return float(np.mean(vals)) if vals else float("nan")


def stats(rows):
    ok = [r["success"] for r in rows]
    centre = [r["centre_mm"] for r in rows if not np.isnan(r["centre_mm"])]
    return {"n": len(rows), "success_rate": float(np.mean(ok)) if ok else float("nan"),
            "auc": auc(rows),
            "median_add_s_mm": float(np.nanmedian([r["add_s_mm"] for r in rows])) if rows else float("nan"),
            "median_centre_mm": float(np.median(centre)) if centre else float("nan"),
            "median_rotation_deg": float(np.nanmedian([r["rotation_deg"] for r in rows])) if rows else float("nan"),
            "errors": sum(bool(r["error"]) for r in rows)}


def tune(args):
    jobs = [("tuning", params, variant, folder, inst, cls, v, d)
            for params, variant in TUNED_ON.items()
            for folder in scene_folders(args.tune)
            for inst, cls in parts_of(folder)
            for v in VOXEL_GRID_MM[cls] for d in TOLERANCE_GRID_MM[cls]]
    path = os.path.join(args.out, "tuning.csv")
    old = []
    if args.extend:                       # keep what is already there, run only the rest
        old = read_rows(path)
        done = {(r["params"], r["scene"], r["instance_id"], r["voxel_mm"], r["tolerance_mm"]) for r in old}
        jobs = [j for j in jobs if (j[1], os.path.basename(j[3]), j[4], j[6], j[7]) not in done]
    print(f"Tuning: {len(jobs)} jobs on {args.tune}" + (f" (+ {len(old)} already done)" if old else ""))
    return old + run_all(jobs, args.workers, path, "tuning", append=bool(old))


def read_rows(path):
    """Rows of a results CSV, with numbers and booleans converted back."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("instance_id", "visible_pixels", "points", "voxel_mm", "tolerance_mm"):
            r[k] = int(float(r[k]))
        for k in ("add_s_mm", "centre_mm", "rotation_deg", "score", "seconds"):
            r[k] = float(r[k])
        r["success"] = r["success"] == "True"
    return rows


def choose(rows, out):
    """Per parameter set and class: the (v, d) with the highest AUC. Prints the grids."""
    chosen = {}
    for params in TUNED_ON:
        chosen[params] = {}
        for cls in VOXEL_GRID_MM:
            print(f"\n{params}, {cls}: AUC / success rate / median centre error (mm), "
                  f"v in rows, d in columns")
            print("  v \\ d " + "".join(f"{d:>21d}" for d in TOLERANCE_GRID_MM[cls]))
            best = None
            for v in VOXEL_GRID_MM[cls]:
                line = f"  {v:5d} "
                for d in TOLERANCE_GRID_MM[cls]:
                    st = stats([r for r in rows if r["params"] == params and r["cls"] == cls
                                and r["voxel_mm"] == v and r["tolerance_mm"] == d])
                    line += f"  {st['auc']:.3f}/{st['success_rate']:.2f}/{st['median_centre_mm']:4.1f}"
                    key = (st["auc"], -st["median_centre_mm"])
                    if best is None or key > best[0]:
                        best = (key, v, d, st)
                print(line)
            _, v, d, st = best
            chosen[params][cls] = {"voxel_mm": v, "tolerance_mm": d, **st}
            edge = (v in (VOXEL_GRID_MM[cls][0], VOXEL_GRID_MM[cls][-1])
                    or d in (TOLERANCE_GRID_MM[cls][0], TOLERANCE_GRID_MM[cls][-1]))
            print(f"  chosen: v = {v} mm, d = {d} mm" + ("   (on the edge of the grid)" if edge else ""))
    with open(os.path.join(out, "chosen.json"), "w") as f:
        json.dump(chosen, f, indent=2)
    return chosen


def evaluate(args, chosen):
    jobs = [("evaluation", params, variant, folder, inst, cls,
             chosen[params][cls]["voxel_mm"], chosen[params][cls]["tolerance_mm"])
            for folder in scene_folders(args.eval)
            for inst, cls in parts_of(folder)
            for variant in VARIANTS for params in TUNED_ON]
    print(f"\nEvaluation: {len(jobs)} jobs on {args.eval}")
    rows = run_all(jobs, args.workers, os.path.join(args.out, "evaluation.csv"), "evaluation")

    summary = {}
    print(f"\n{'params':15s} {'variant':10s} {'class':8s} {'n':>4s} {'AUC':>6s} {'success':>8s} "
          f"{'ADD-S':>6s} {'centre':>7s} {'rot':>6s}  (medians, mm / deg)")
    for params in TUNED_ON:
        for variant in VARIANTS:
            for cls in ["all"] + list(VOXEL_GRID_MM):
                sel = [r for r in rows if r["params"] == params and r["variant"] == variant
                       and (cls == "all" or r["cls"] == cls)]
                s = stats(sel)
                summary[f"{params}|{variant}|{cls}"] = s
                print(f"{params:15s} {variant:10s} {cls:8s} {s['n']:4d} {s['auc']:6.3f} {s['success_rate']:8.3f} "
                      f"{s['median_add_s_mm']:6.2f} {s['median_centre_mm']:7.2f} {s['median_rotation_deg']:6.1f}")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    a, b, c = (summary["tuned_on_ideal|ideal|all"], summary["tuned_on_ideal|realistic|all"],
               summary["tuned_on_realistic|realistic|all"])
    for name in ("auc", "success_rate"):
        print(f"{name:12s}  A (ideal -> ideal): {a[name]:.3f}   B (ideal -> realistic): {b[name]:.3f}   "
              f"C (realistic -> realistic): {c[name]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tune", default="outputs/dataset", help="tuning dataset")
    parser.add_argument("--eval", default="outputs/dataset_eval", help="evaluation dataset")
    parser.add_argument("--out", default="outputs/experiment", help="output folder")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true", help="reuse chosen.json")
    parser.add_argument("--rechoose", action="store_true", help="choose again from tuning.csv")
    parser.add_argument("--extend", action="store_true", help="run only grid points missing from tuning.csv")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for cls, make in MAKE_PART.items():
        DIAMETER_MM[cls] = 1000 * diameter(make())
    if args.skip_tuning:
        with open(os.path.join(args.out, "chosen.json")) as f:
            chosen = json.load(f)
    else:
        rows = read_rows(os.path.join(args.out, "tuning.csv")) if args.rechoose else tune(args)
        chosen = choose(rows, args.out)
    if not args.tune_only:
        evaluate(args, chosen)
