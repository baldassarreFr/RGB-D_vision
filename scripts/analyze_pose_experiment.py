"""M6: analysis of the pose experiment (reads evaluation.csv, writes tables and figures).

Usage: python scripts/analyze_pose_experiment.py [--exp outputs/experiment]
Conditions (rows of evaluation.csv):
    A = tuned on ideal, evaluated on ideal        (what ideal synthetic data promise)
    B = tuned on ideal, evaluated on realistic depth     (same pipeline and parameters)
    C = tuned on realistic, evaluated on realistic depth (camera modelled during development)
    (realistic = all effects of the D435i camera model)
Writes in --exp:
    analysis.txt            everything printed below
    analysis.json           the same numbers, machine-readable
    results_table.tex       LaTeX rows for the report (A, B, C and the ablation)
Figures (also copied to --figdir, default docs/figures, for the report and the README):
    fig_accuracy_centre.png share of parts with centre error below a threshold, A/B/C, per class
    fig_accuracy_adds.png   the same for ADD-S, up to the success threshold
    fig_conditions.png      AUC and median centre error of A, B, C with 95 % CI
    fig_ablation.png        the same for each single camera effect (parameters tuned on ideal)
Statistics:
    95 % confidence intervals by bootstrap over SCENES (parts of one scene are not independent):
    2000 resamples of the evaluation scenes with replacement.
    Paired comparison (same part in two conditions): how many parts change from success to
    failure and back; exact McNemar test on those changes; median paired change of the centre
    error.
"""
import argparse
import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from rgbd_sim.metrics import diameter
from rgbd_sim.parts import make_bracket, make_profile

N_BOOT = 2000
CONDITIONS = {"A": ("tuned_on_ideal", "ideal"), "B": ("tuned_on_ideal", "realistic"),
              "C": ("tuned_on_realistic", "realistic")}
ABLATION = ["ideal", "noise", "aluminium", "shadows", "edges", "realistic"]   # tuned on ideal
CLASSES = ["all", "bracket", "profile"]
DIAMETER_MM = {"profile": 1000 * diameter(make_profile()), "bracket": 1000 * diameter(make_bracket())}


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("add_s_mm", "centre_mm", "rotation_deg", "score", "seconds"):
            r[k] = float(r[k])
        r["success"] = r["success"] == "True"
        r["key"] = (r["scene"], r["instance_id"])
        # AUC contribution: 1 at ADD-S 0, 0 at the success threshold (no estimate -> 0)
        r["auc"] = 0.0 if math.isnan(r["add_s_mm"]) else \
            max(0.0, 1.0 - r["add_s_mm"] / (0.1 * DIAMETER_MM[r["cls"]]))
    return rows


def select(rows, params, variant, cls="all"):
    return [r for r in rows if r["params"] == params and r["variant"] == variant
            and (cls == "all" or r["cls"] == cls)]


# ---------- statistics ----------
STRICT = {"adds_below_1mm": ("add_s_mm", 1.0), "adds_below_2mm": ("add_s_mm", 2.0),
          "centre_below_1mm": ("centre_mm", 1.0), "centre_below_2mm": ("centre_mm", 2.0),
          "centre_below_5mm": ("centre_mm", 5.0)}


def measures(rows):
    """The numbers reported for a set of rows (one per part)."""
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
    """95 % CI of fn(rows) by resampling scenes with replacement."""
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


CI_KEYS = ("success", "auc", "median_centre_mm") + tuple(STRICT)


def with_ci(rows, seed=0):
    """measures(rows) plus a 95 % CI for each key in CI_KEYS (one bootstrap over scenes for all)."""
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
    """Same parts in two conditions: changes of success, McNemar test, paired centre change."""
    a = {r["key"]: r for r in rows1}
    b = {r["key"]: r for r in rows2}
    keys = sorted(set(a) & set(b))
    lost = sum(a[k]["success"] and not b[k]["success"] for k in keys)
    gained = sum(not a[k]["success"] and b[k]["success"] for k in keys)
    n = lost + gained          # exact McNemar: two-sided binomial test on the changed parts
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
parser.add_argument("--exp", default="outputs/experiment")
parser.add_argument("--figdir", default="docs/figures", help="figures for the report and README (in git)")
args = parser.parse_args()
rows = read(os.path.join(args.exp, "evaluation.csv"))
out_lines, result = [], {}


def say(text=""):
    print(text)
    out_lines.append(text)


fmt = lambda m, k, p=3: f"{m[k]:.{p}f} [{m[k + '_ci'][0]:.{p}f}, {m[k + '_ci'][1]:.{p}f}]"
say(f"Evaluation: {len({r['key'] for r in rows})} parts in {len({r['scene'] for r in rows})} scenes; "
    f"success = ADD-S < 10 % of the diameter ({DIAMETER_MM['profile']:.1f} / "
    f"{DIAMETER_MM['bracket']:.1f} mm -> {0.1 * DIAMETER_MM['profile']:.1f} / {0.1 * DIAMETER_MM['bracket']:.1f} mm)")
say("95 % CI by bootstrap over scenes.\n")

say("1. Conditions A, B, C")
result["conditions"] = {}
for cls in CLASSES:
    say(f"  {cls}")
    for name, (params, variant) in CONDITIONS.items():
        m = with_ci(select(rows, params, variant, cls))
        result["conditions"][f"{name}|{cls}"] = m
        say(f"    {name}  n={m['n']:3d}  success {fmt(m, 'success')}  AUC {fmt(m, 'auc')}  "
            f"centre {fmt(m, 'median_centre_mm', 2)} mm  ADD-S {m['median_add_s_mm']:.2f} mm  "
            f"rot {m['median_rotation_deg']:.1f} deg  no estimate {m['no_estimate']}")

say("\n1b. Stricter thresholds: share of parts below an absolute error (the standard success")
say("    threshold is 16.0 mm for profiles and 6.7 mm for brackets)")
for cls in CLASSES:
    say(f"  {cls}")
    for name in CONDITIONS:
        m = result["conditions"][f"{name}|{cls}"]
        say(f"    {name}  " + "  ".join(f"{k.replace('_below_', ' < ')} {fmt(m, k, 2)}" for k in STRICT))

say("\n2. Paired comparisons (same parts)")
result["paired"] = {}
for (x, y) in (("A", "B"), ("B", "C"), ("A", "C")):
    for cls in CLASSES:
        p = paired(select(rows, *CONDITIONS[x], cls), select(rows, *CONDITIONS[y], cls))
        result["paired"][f"{x}->{y}|{cls}"] = p
        say(f"  {x} -> {y} {cls:8s}: success->failure {p['success_to_failure']:3d}, "
            f"failure->success {p['failure_to_success']:3d} (McNemar p = {p['mcnemar_p']:.2g}); "
            f"AUC change {p['auc_change']:+.3f} [{p['auc_change_ci'][0]:+.3f}, {p['auc_change_ci'][1]:+.3f}]; "
            f"median centre change {p['median_centre_change_mm']:+.2f} mm "
            f"[{p['median_centre_change_ci'][0]:+.2f}, {p['median_centre_change_ci'][1]:+.2f}]")

say("\n3. Ablation: parameters tuned on ideal, one camera effect at a time")
result["ablation"] = {}
for cls in CLASSES:
    say(f"  {cls}")
    ideal_rows = select(rows, "tuned_on_ideal", "ideal", cls)
    for variant in ABLATION:
        sel = select(rows, "tuned_on_ideal", variant, cls)
        m = with_ci(sel)
        p = paired(ideal_rows, sel)
        result["ablation"][f"{variant}|{cls}"] = {**m, "vs_ideal": p}
        say(f"    {variant:10s} success {fmt(m, 'success')}  AUC {fmt(m, 'auc')}  "
            f"centre {fmt(m, 'median_centre_mm', 2)} mm  | vs ideal: lost {p['success_to_failure']}, "
            f"gained {p['failure_to_success']}, AUC change {p['auc_change']:+.3f}")

say("\n4. Parameters tuned on realistic, evaluated on ideal (for completeness)")
for cls in CLASSES:
    m = measures(select(rows, "tuned_on_realistic", "ideal", cls))
    say(f"  {cls:8s} success {m['success']:.3f}  AUC {m['auc']:.3f}  centre {m['median_centre_mm']:.2f} mm")

say(f"\nMean time per part: {np.mean([r['seconds'] for r in rows if r['seconds'] > 0]):.2f} s")

# ---------- files ----------
with open(os.path.join(args.exp, "analysis.txt"), "w") as f:
    f.write("\n".join(out_lines) + "\n")
with open(os.path.join(args.exp, "analysis.json"), "w") as f:
    json.dump(result, f, indent=2)

# LaTeX rows: condition / variant, success, AUC, median centre error (all parts)
tex = ["% generated by scripts/analyze_pose_experiment.py",
       "% condition & success & AUC & median centre error (mm) \\\\"]
for name in CONDITIONS:
    m = result["conditions"][f"{name}|all"]
    tex.append(f"{name} & {100 * m['success']:.1f}\\,\\% & {m['auc']:.3f} & {m['median_centre_mm']:.2f} \\\\")
tex.append("% ablation (tuned on ideal): variant & success & AUC & median centre error (mm) \\\\")
for variant in ABLATION:
    m = result["ablation"][f"{variant}|all"]
    tex.append(f"{variant} & {100 * m['success']:.1f}\\,\\% & {m['auc']:.3f} & {m['median_centre_mm']:.2f} \\\\")
tex.append("% stricter thresholds, all parts: condition & ADD-S<1mm & ADD-S<2mm & centre<1mm & centre<2mm & centre<5mm \\\\")
for name in CONDITIONS:
    m = result["conditions"][f"{name}|all"]
    tex.append(f"{name} & " + " & ".join(f"{100 * m[k]:.1f}\\,\\%" for k in STRICT) + " \\\\")
with open(os.path.join(args.exp, "results_table.tex"), "w") as f:
    f.write("\n".join(tex) + "\n")

# ---------- figures (for the report and the README) ----------
# Colours: validated categorical palette (slots 1-3), plus line style / marker as second cue.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COND_STYLE = {"A": dict(color="#2a78d6", ls="-", label="A  tuned on ideal, tested on ideal"),
              "B": dict(color="#eb6834", ls="-", label="B  tuned on ideal, tested on realistic"),
              "C": dict(color="#1baf7a", ls="--", label="C  tuned on realistic, tested on realistic")}
CLASS_STYLE = {"bracket": dict(color="#2a78d6", marker="o"), "profile": dict(color="#eb6834", marker="s")}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
os.makedirs(args.figdir, exist_ok=True)


def save(fig, name):
    for folder in {args.exp, args.figdir}:
        fig.savefig(os.path.join(folder, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def curves(key, xmax_of, xlabel, name, threshold=False):
    """Share of parts whose error is below t, as t grows: one panel per class, A/B/C."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, cls in zip(axes, ["bracket", "profile"]):
        xmax = xmax_of(cls)
        t = np.linspace(0, xmax, 400)
        for name_c, (params, variant) in CONDITIONS.items():
            vals = np.array([r[key] for r in select(rows, params, variant, cls)])
            vals = np.where(np.isnan(vals), np.inf, vals)
            ax.plot(t, [(vals < x).mean() for x in t], lw=2, **COND_STYLE[name_c])
        if threshold:
            ax.axvline(xmax, color=INK2, lw=1, ls=":")
            ax.text(xmax, 0.03, "success\nthreshold ", ha="right", va="bottom", color=INK2, fontsize=8)
        ax.set_title(f"{cls}s (n = {len(select(rows, 'tuned_on_ideal', 'ideal', cls))})", color=INK)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.02)
        ax.grid(color=GRID, lw=0.8)
    axes[0].set_ylabel("share of parts with error below threshold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    save(fig, name)


curves("centre_mm", lambda cls: 10.0, "centre error threshold [mm]", "fig_accuracy_centre.png")
curves("add_s_mm", lambda cls: 0.1 * DIAMETER_MM[cls], "ADD-S threshold [mm]", "fig_accuracy_adds.png",
       threshold=True)


def dot_panels(entries, labels, name, title):
    """Rows = labels; two panels (ADD-S AUC, median centre error) with 95 % CI, one marker per class."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(labels) + 1.4), sharey=True)
    y = np.arange(len(labels))[::-1]
    for ax, (metric, xlabel) in zip(axes, (("auc", "ADD-S AUC (higher is better)"),
                                           ("median_centre_mm", "median centre error [mm] (lower is better)"))):
        for k, cls in enumerate(["bracket", "profile"]):
            ms = [entries(lab, cls) for lab in labels]
            x = np.array([m[metric] for m in ms])
            err = np.array([[m[metric] - m[metric + "_ci"][0], m[metric + "_ci"][1] - m[metric]] for m in ms]).T
            ax.errorbar(x, y + (0.12 if k == 0 else -0.12), xerr=err, fmt=CLASS_STYLE[cls]["marker"],
                        color=CLASS_STYLE[cls]["color"], ms=7, capsize=3, lw=1.5, label=f"{cls}s")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color=GRID, lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[1].set_xlim(left=0)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, color=INK, y=1.07)
    save(fig, name)


dot_panels(lambda lab, cls: result["conditions"][f"{lab[0]}|{cls}"],
           ["A  ideal -> ideal", "B  ideal -> realistic", "C  realistic -> realistic"],
           "fig_conditions.png", "Pose estimation in the three conditions (95 % CI over scenes)")
ABLATION_LABELS = {"ideal": "ideal depth", "shadows": "+ stereo shadows only", "edges": "+ edges only",
                   "noise": "+ noise (matte level) only", "aluminium": "+ noise x2.9 (aluminium) only",
                   "realistic": "realistic (all effects)"}
dot_panels(lambda lab, cls: result["ablation"][f"{[k for k, v in ABLATION_LABELS.items() if v == lab][0]}|{cls}"],
           list(ABLATION_LABELS.values()), "fig_ablation.png",
           "Parameters tuned on ideal depth, one camera effect at a time (95 % CI over scenes)")
print(f"\nwritten in {args.exp}: analysis.txt, analysis.json, results_table.tex; figures also in {args.figdir}:"
      f" fig_accuracy_centre.png, fig_accuracy_adds.png, fig_conditions.png, fig_ablation.png")
