"""One-off: rename the camera variant "full" to "realistic" in the files already produced.

Usage:
    python scripts/rename_full_to_realistic.py            # dry run: only prints what it would do
    python scripts/rename_full_to_realistic.py --apply    # does it
What it changes (nothing else is touched):
  outputs/dataset*/scenes/*/depth_full.png      -> depth_realistic.png      (renamed, same bytes)
  outputs/dataset*/camera_variants_summary.json key "full" -> "realistic"
  outputs/experiment/tuning.csv, evaluation.csv column "variant": full -> realistic,
                                                column "params":  tuned_on_full -> tuned_on_realistic
  outputs/experiment/chosen.json                key tuned_on_full -> tuned_on_realistic
  outputs/experiment/summary.json               keys "tuned_on_full|full|..." -> "tuned_on_realistic|realistic|..."
  outputs/experiment/analysis.txt               kept as analysis_before_rename.txt, to compare after
                                                re-running scripts/analyze_pose_experiment.py
Checks after --apply: same number of rows, every other column identical, no "full" left.
"""
import argparse
import csv
import glob
import json
import os
import shutil

OLD, NEW = "full", "realistic"
OLD_P, NEW_P = "tuned_on_full", "tuned_on_realistic"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--apply", action="store_true", help="actually change the files")
parser.add_argument("--outputs", default="outputs")
args = parser.parse_args()
act = "RENAME" if args.apply else "would rename"


def swap(value):
    return {OLD: NEW, OLD_P: NEW_P}.get(value, value)


# ---------- 1. Depth images and variant summaries ----------
for dataset in sorted(glob.glob(os.path.join(args.outputs, "dataset*"))):
    files = sorted(glob.glob(os.path.join(dataset, "scenes", "*", f"depth_{OLD}.png")))
    clash = [f for f in files if os.path.exists(f.replace(f"depth_{OLD}.png", f"depth_{NEW}.png"))]
    if clash:
        raise SystemExit(f"stop: {len(clash)} files depth_{NEW}.png already exist in {dataset}")
    print(f"{dataset}: {act} {len(files)} files depth_{OLD}.png -> depth_{NEW}.png")
    if args.apply:
        for f in files:
            os.rename(f, f.replace(f"depth_{OLD}.png", f"depth_{NEW}.png"))
    path = os.path.join(dataset, "camera_variants_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            summary = json.load(f)
        print(f"  {path}: key '{OLD}' -> '{NEW}'" + ("" if OLD in summary else "  (not present)"))
        if args.apply and OLD in summary:
            summary = {swap(k): v for k, v in summary.items()}
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)

# ---------- 2. Experiment results ----------
exp = os.path.join(args.outputs, "experiment")
for name in ("tuning.csv", "evaluation.csv"):
    path = os.path.join(exp, name)
    if not os.path.exists(path):
        continue
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    n_var = sum(r["variant"] == OLD for r in rows)
    n_par = sum(r["params"] == OLD_P for r in rows)
    print(f"{path}: {len(rows)} rows; {act} variant in {n_var} rows, params in {n_par} rows")
    if args.apply:
        new_rows = [dict(r, variant=swap(r["variant"]), params=swap(r["params"])) for r in rows]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(new_rows)
        # check: read back, same rows, only the two columns changed, no OLD left
        with open(path, newline="") as f:
            back = list(csv.DictReader(f))
        assert len(back) == len(rows), "row count changed"
        for a, b in zip(rows, back):
            for k in fields:
                if k in ("variant", "params"):
                    assert b[k] == swap(a[k])
                else:
                    assert b[k] == a[k], f"column {k} changed"
        assert not any(r["variant"] == OLD or r["params"] == OLD_P for r in back)
        print(f"  checked: {len(back)} rows, all other columns identical, no '{OLD}' left")

for name in ("chosen.json", "summary.json"):
    path = os.path.join(exp, name)
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)
    renamed = {"|".join(swap(p) for p in k.split("|")): v for k, v in data.items()}
    changed = sum(a != b for a, b in zip(data, renamed))
    print(f"{path}: {act} {changed} keys")
    if args.apply:
        assert list(renamed.values()) == list(data.values())
        with open(path, "w") as f:
            json.dump(renamed, f, indent=2)

path = os.path.join(exp, "analysis.txt")
backup = os.path.join(exp, "analysis_before_rename.txt")
if os.path.exists(path) and not os.path.exists(backup):        # never overwrite the first copy
    print(f"{path}: {'copied' if args.apply else 'would copy'} to analysis_before_rename.txt")
    if args.apply:
        shutil.copy(path, backup)

if not args.apply:
    print("\nDry run: nothing changed. Run again with --apply.")
else:
    print("\nDone. Now run: python scripts/analyze_pose_experiment.py  and compare analysis.txt with "
          "analysis_before_rename.txt (same numbers, new names).")
