"""
Step 2 analysis: the pose experiment with two camera effects that have no published value.

Usage: python scripts/analyze_step2.py [--exp outputs/step2] [--step1 results/step1] [--figdir docs/figures]

Writes in --exp: analysis.txt, analysis.json, results_table.tex

Figures (in --exp and --figdir):
    fig_step2_B.png   parameters tuned on ideal: Step 1 realistic and each Step 2 variant
    fig_step2_C.png   parameters tuned on the depth they are tested on: the same rows
"""
import argparse
import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from rgbd_sim.camera_model import STEP2_VARIANTS
from rgbd_sim.metrics import diameter
from rgbd_sim.parts import make_bracket, make_profile

N_BOOT = 2000
CLASSES = ["all", "bracket", "profile"]
DIAMETER_MM = {"profile": 1000 * diameter(make_profile()), "bracket": 1000 * diameter(make_bracket())}
STEP1 = {"A": ("tuned_on_ideal", "ideal"), "B realistic": ("tuned_on_ideal", "realistic"),
         "C realistic": ("tuned_on_realistic", "realistic")}


def label_of(variant):
    """'realistic_holes45' -> '+ holes (aluminium > 45 deg)', 'realistic_corr5' -> '+ correlated noise (5 px)'."""
    effect = variant.replace("realistic_", "")
    if effect.startswith("holes"):
        return f"+ holes (aluminium > {effect[5:]} deg)"
    return f"+ correlated noise ({effect[4:]} px)"


# ---------- reading and statistics ----------
def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("add_s_mm", "centre_mm", "rotation_deg", "score", "seconds"):
            r[k] = float(r[k])
        r["success"] = r["success"] == "True"
        r["key"] = (r["scene"], r["instance_id"])
        r["auc"] = 0.0 if math.isnan(r["add_s_mm"]) else \
            max(0.0, 1.0 - r["add_s_mm"] / (0.1 * DIAMETER_MM[r["cls"]]))
    return rows


def select(rows, params, variant, cls="all"):
    return [r for r in rows if r["params"] == params and r["variant"] == variant
            and (cls == "all" or r["cls"] == cls)]


STRICT = {"adds_below_1mm": ("add_s_mm", 1.0), "adds_below_2mm": ("add_s_mm", 2.0),
          "centre_below_1mm": ("centre_mm", 1.0), "centre_below_2mm": ("centre_mm", 2.0),
          "centre_below_5mm": ("centre_mm", 5.0)}
CI_KEYS = ("success", "auc", "median_centre_mm") + tuple(STRICT)


def measures(rows):
    centre = np.array([r["centre_mm"] for r in rows])
    strict = {name: float(np.mean([(r[key] < t) if not math.isnan(r[key]) else False for r in rows]))
              for name, (key, t) in STRICT.items()}
    return {"n": len(rows), **strict,
            "success": float(np.mean([r["success"] for r in rows])),
            "auc": float(np.mean([r["auc"] for r in rows])),
            "median_add_s_mm": float(np.nanmedian([r["add_s_mm"] for r in rows])),
            "median_centre_mm": float(np.nanmedian(centre)),
            "median_rotation_deg": float(np.nanmedian([r["rotation_deg"] for r in rows])),
            "no_estimate": int(np.isnan(centre).sum())}


def bootstrap(rows, fn, seed=0):
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r["scene"], []).append(r)
    scenes = list(by_scene)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(scenes), len(scenes))
        vals.append(fn([r for i in pick for r in by_scene[scenes[i]]]))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def with_ci(rows, seed=0):
    m = measures(rows)
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r["scene"], []).append(r)
    scenes = list(by_scene)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(scenes), len(scenes))
        samples.append(measures([r for i in pick for r in by_scene[scenes[i]]]))
    for k in CI_KEYS:
        vals = [x[k] for x in samples]
        m[k + "_ci"] = [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
    return m


def paired(rows1, rows2):
    a = {r["key"]: r for r in rows1}
    b = {r["key"]: r for r in rows2}
    keys = sorted(set(a) & set(b))
    lost = sum(a[k]["success"] and not b[k]["success"] for k in keys)
    gained = sum(not a[k]["success"] and b[k]["success"] for k in keys)
    n = lost + gained
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(lost, gained) + 1)) / 2 ** n)
    diff_rows = [{"scene": k[0], "d": b[k]["centre_mm"] - a[k]["centre_mm"],
                  "dauc": b[k]["auc"] - a[k]["auc"]} for k in keys]
    med = lambda rr: float(np.nanmedian([r["d"] for r in rr]))
    mean_auc = lambda rr: float(np.mean([r["dauc"] for r in rr]))
    return {"parts": len(keys), "success_to_failure": lost, "failure_to_success": gained,
            "mcnemar_p": p,
            "median_centre_change_mm": med(diff_rows), "median_centre_change_ci": bootstrap(diff_rows, med),
            "auc_change": mean_auc(diff_rows), "auc_change_ci": bootstrap(diff_rows, mean_auc)}


# ---------- main ----------
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--exp", default="outputs/step2")
parser.add_argument("--step1", default="results/step1", help="Step 1 results (read only)")
parser.add_argument("--figdir", default="docs/figures")
args = parser.parse_args()
step1 = read(os.path.join(args.step1, "evaluation.csv"))
step2 = read(os.path.join(args.exp, "evaluation.csv"))
with open(os.path.join(args.exp, "chosen.json")) as f:
    chosen = json.load(f)
variants = [v for v in STEP2_VARIANTS if any(r["variant"] == v for r in step2)]
out_lines, result = [], {"step1": {}, "B": {}, "C": {}, "paired": {}}


def say(text=""):
    print(text)
    out_lines.append(text)


fmt = lambda m, k, p=3: f"{m[k]:.{p}f} [{m[k + '_ci'][0]:.{p}f}, {m[k + '_ci'][1]:.{p}f}]"


def line(name, m):
    return (f"    {name:36s} n={m['n']:3d}  success {fmt(m, 'success')}  AUC {fmt(m, 'auc')}  "
            f"centre {fmt(m, 'median_centre_mm', 2)} mm  no estimate {m['no_estimate']}")


say(f"Step 2: {len(variants)} variants; {len({r['key'] for r in step2})} parts in "
    f"{len({r['scene'] for r in step2})} scenes. Step 1 references read from {args.step1}.")
say("95 % CI by bootstrap over scenes.\n")

say("0. Parameters (v = voxel, d = tolerance, mm) and their AUC on the tuning scenes")
for params in ["tuned_on_ideal"] + [f"tuned_on_{v}" for v in variants]:
    c = chosen[params]
    say(f"  {params:32s} " + "   ".join(f"{cls}: v = {c[cls]['voxel_mm']}, d = {c[cls]['tolerance_mm']}"
                                        f" (AUC {c[cls]['auc']:.3f})" for cls in ("bracket", "profile")))

say("\n1. Success, AUC, median centre error")
for cls in CLASSES:
    say(f"  {cls}")
    for name, (params, variant) in STEP1.items():
        m = with_ci(select(step1, params, variant, cls))
        result["step1"][f"{name}|{cls}"] = m
        say(line(f"Step 1  {name}", m))
    for v in variants:
        for cond, params in (("B", "tuned_on_ideal"), ("C", f"tuned_on_{v}")):
            m = with_ci(select(step2, params, v, cls))
            result[cond][f"{v}|{cls}"] = m
            say(line(f"{cond}  {label_of(v)}", m))

say("\n2. Stricter thresholds: share of parts below an absolute error")
for cls in CLASSES:
    say(f"  {cls}")
    entries = [(f"Step 1  {n}", result["step1"][f"{n}|{cls}"]) for n in STEP1] + \
              [(f"{c}  {label_of(v)}", result[c][f"{v}|{cls}"]) for v in variants for c in ("B", "C")]
    for name, m in entries:
        say(f"    {name:36s} " + "  ".join(f"{k.replace('_below_', ' < ')} {fmt(m, k, 2)}" for k in STRICT))

say("\n3. Paired comparisons (same parts)")
for v in variants:
    say(f"  {label_of(v)}")
    for (x_name, x_rows, x_sel), (y_name, y_sel) in (
            (("B realistic (Step 1)", step1, ("tuned_on_ideal", "realistic")), ("B variant", ("tuned_on_ideal", v))),
            (("B variant", step2, ("tuned_on_ideal", v)), ("C variant", (f"tuned_on_{v}", v)))):
        for cls in CLASSES:
            p = paired(select(x_rows, *x_sel, cls), select(step2, *y_sel, cls))
            result["paired"][f"{v}|{x_name} -> {y_name}|{cls}"] = p
            say(f"    {x_name} -> {y_name} {cls:8s}: success->failure {p['success_to_failure']:3d}, "
                f"failure->success {p['failure_to_success']:3d} (McNemar p = {p['mcnemar_p']:.2g}); "
                f"AUC change {p['auc_change']:+.3f} [{p['auc_change_ci'][0]:+.3f}, {p['auc_change_ci'][1]:+.3f}]; "
                f"median centre change {p['median_centre_change_mm']:+.2f} mm "
                f"[{p['median_centre_change_ci'][0]:+.2f}, {p['median_centre_change_ci'][1]:+.2f}]")

# ---------- files ----------
with open(os.path.join(args.exp, "analysis.txt"), "w") as f:
    f.write("\n".join(out_lines) + "\n")
with open(os.path.join(args.exp, "analysis.json"), "w") as f:
    json.dump(result, f, indent=2)
tex = ["% generated by scripts/analyze_step2.py (all parts)",
       "% row & B: success & B: AUC & B: centre (mm) & B: centre<2mm & C: success & C: AUC & C: centre (mm) & C: centre<2mm \\\\"]
rows_tex = [("realistic (Step 1)", result["step1"]["B realistic|all"], result["step1"]["C realistic|all"])] + \
           [(label_of(v), result["B"][f"{v}|all"], result["C"][f"{v}|all"]) for v in variants]
for name, b, c in rows_tex:
    cells = [f"{100 * m['success']:.1f}\\,\\% & {m['auc']:.3f} & {m['median_centre_mm']:.2f} & "
             f"{100 * m['centre_below_2mm']:.0f}\\,\\%" for m in (b, c)]
    name = name.replace(">", "$>$").replace(" deg", "$^\\circ$")
    tex.append(f"{name} & {cells[0]} & {cells[1]} \\\\")
with open(os.path.join(args.exp, "results_table.tex"), "w") as f:
    f.write("\n".join(tex) + "\n")

# ---------- figures ----------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLASS_STYLE = {"bracket": dict(color="#2a78d6", marker="o"), "profile": dict(color="#eb6834", marker="s")}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
os.makedirs(args.figdir, exist_ok=True)

def figure_labels(cond):
    """Row labels; where parts have no estimate (< 10 measured points), how many, on a second line."""
    labels = ["realistic (Step 1)"]
    for v in variants:
        m = result[cond][f"{v}|all"]
        labels.append(label_of(v).replace(" deg", "°")
                      + (f"\n{m['no_estimate']} of {m['n']} parts: no estimate" if m["no_estimate"] else ""))
    return labels


# same x range in both figures, so that B and C can be compared by eye (only the plotted rows)
all_m = [m for key in ("B", "C") for m in result[key].values()] + \
        [m for k, m in result["step1"].items() if not k.startswith("A|")]
xlim = {"auc": (min(m["auc_ci"][0] for m in all_m) - 0.02, 1.0),
        # log axis: errors from ~1 mm to ~30 mm; on a linear axis the small ones are squashed together
        "median_centre_mm": (min(1.0, min(m["median_centre_mm_ci"][0] for m in all_m) * 0.9),
                             max(m["median_centre_mm_ci"][1] for m in all_m) * 1.15)}
CENTRE_TICKS = [1, 2, 5, 10, 20, 50]


def dot_panels(entries, labels, name, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(labels) + 1.4), sharey=True)
    y = np.arange(len(labels))[::-1]
    for ax, (metric, xlabel) in zip(axes, (("auc", "ADD-S AUC (higher is better)"),
                                           ("median_centre_mm", "median centre error [mm], log scale\n"
                                            "(parts with an estimate; lower is better)"))):
        if metric == "median_centre_mm":
            ax.set_xscale("log")
            lo, hi = xlim[metric]
            ax.set_xticks([t for t in CENTRE_TICKS if lo <= t <= hi])
            ax.set_xticklabels([str(t) for t in CENTRE_TICKS if lo <= t <= hi])
            ax.minorticks_off()
        for k, cls in enumerate(["bracket", "profile"]):
            ms = [entries(i, cls) for i in range(len(labels))]
            x = np.array([m[metric] for m in ms])
            err = np.array([[m[metric] - m[metric + "_ci"][0], m[metric + "_ci"][1] - m[metric]] for m in ms]).T
            ax.errorbar(x, y + (0.12 if k == 0 else -0.12), xerr=err, fmt=CLASS_STYLE[cls]["marker"],
                        color=CLASS_STYLE[cls]["color"], ms=7, capsize=3, lw=1.5, label=f"{cls}s")
        ax.set_xlim(*xlim[metric])
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color=GRID, lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, color=INK, y=1.07)
    for folder in {args.exp, args.figdir}:
        fig.savefig(os.path.join(folder, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


dot_panels(lambda i, cls: result["step1"][f"B realistic|{cls}"] if i == 0 else result["B"][f"{variants[i - 1]}|{cls}"],
           figure_labels("B"), "fig_step2_B.png", "Step 2, parameters tuned on ideal depth (B), 95 % CI over scenes")
dot_panels(lambda i, cls: result["step1"][f"C realistic|{cls}"] if i == 0 else result["C"][f"{variants[i - 1]}|{cls}"],
           figure_labels("C"), "fig_step2_C.png", "Step 2, parameters tuned on the tested depth (C), 95 % CI over scenes")
print(f"\nwritten in {args.exp}: analysis.txt, analysis.json, results_table.tex; "
      f"figures also in {args.figdir}: fig_step2_B.png, fig_step2_C.png")
