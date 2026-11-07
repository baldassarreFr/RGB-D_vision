"""Analysis of the part/table segmentation experiment (numbers and figures, no interpretation).

Usage: python scripts/analyze_segment_experiment.py [--exp outputs/segmentation] [--figdir docs/figures]
Conditions (parameter set, variant):
    A = tuned_on_ideal (t chosen on ideal depth) on ideal depth
    B = tuned_on_ideal on realistic depth; on ideal depth t = 1, 2, 3 mm were equally perfect, so B is
        also given for the other tied values (range), with the middle one as reference
    C = tuned_on_realistic (t chosen on realistic depth) on realistic depth
Measures (pooled over parts or pixels of the selected scenes):
    found50 / found75  share of parts with >= 50 % / >= 75 % of their pixels marked "part"
    recall             mean share of a part's pixels marked "part"
    precision          share of the pixels marked "part" that belong to a part (all classes together)
    F1                 harmonic mean of found50 and precision
95 % CI: 2000 resamples of the evaluation scenes. Paired comparison on the same parts: how many
change from found to missed and back (exact McNemar test).
Writes in --exp: analysis.txt, analysis.json, results_table.tex; figures in --exp and --figdir:
    fig_seg_threshold.png   tuning scenes: found50, precision, F1 against t, ideal and realistic
    fig_seg_conditions.png  A, B (with the tied values), C: found50 per class and precision
    fig_seg_ablation.png    parameters tuned on ideal, one camera effect at a time
"""
import argparse
import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

N_BOOT = 2000
VARIANTS = ["ideal", "noise", "aluminium", "shadows", "edges", "realistic"]
CLASSES = ["all", "bracket", "profile"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--exp", default="outputs/segmentation")
parser.add_argument("--figdir", default="docs/figures")
args = parser.parse_args()
os.makedirs(args.figdir, exist_ok=True)


def read(name):
    with open(os.path.join(args.exp, name), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("found50", "found75"):
                r[k] = v == "True"
            elif k in ("recall", "precision", "plane_angle_deg", "plane_offset_mm"):
                r[k] = float(v)
            elif k in ("t_mm", "instance_id", "part_pixels_marked", "correct_pixels_marked"):
                r[k] = int(float(v))
    return rows


parts, scenes = read("evaluation_parts.csv"), read("evaluation_scenes.csv")
tparts, tscenes = read("tuning_parts.csv"), read("tuning_scenes.csv")
with open(os.path.join(args.exp, "chosen.json")) as f:
    chosen = json.load(f)
t_ideal, t_real = chosen["tuned_on_ideal"]["t_mm"], chosen["tuned_on_realistic"]["t_mm"]
ties = [t for t in chosen["tuned_on_ideal"].get("ties_mm", [t_ideal]) if t != t_ideal]
SETS = {"tuned_on_ideal": t_ideal, **{f"tuned_on_ideal_tie_t{t}": t for t in ties},
        "tuned_on_realistic": t_real}
CONDITIONS = {"A": ("tuned_on_ideal", "ideal"), "B": ("tuned_on_ideal", "realistic"),
              "C": ("tuned_on_realistic", "realistic")}


def select(rows, params, variant, cls="all"):
    return [r for r in rows if r["params"] == params and r["variant"] == variant
            and (cls == "all" or r.get("cls") == cls)]


def measures(p_rows, s_rows):
    found = float(np.mean([r["found50"] for r in p_rows])) if p_rows else float("nan")
    marked = sum(r["part_pixels_marked"] for r in s_rows)
    precision = sum(r["correct_pixels_marked"] for r in s_rows) / marked if marked else 0.0
    return {"n_parts": len(p_rows), "found50": found,
            "found75": float(np.mean([r["found75"] for r in p_rows])) if p_rows else float("nan"),
            "recall": float(np.mean([r["recall"] for r in p_rows])) if p_rows else float("nan"),
            "precision": precision,
            "f1": 0.0 if found + precision == 0 else 2 * found * precision / (found + precision)}


def with_ci(p_rows, s_rows, seed=0):
    m = measures(p_rows, s_rows)
    names = sorted({r["scene"] for r in s_rows})
    bp = {n: [r for r in p_rows if r["scene"] == n] for n in names}
    bs = {n: [r for r in s_rows if r["scene"] == n] for n in names}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(N_BOOT):
        pick = [names[i] for i in rng.choice(len(names), len(names))]
        samples.append(measures([r for n in pick for r in bp[n]], [r for n in pick for r in bs[n]]))
    for k in ("found50", "found75", "recall", "precision", "f1"):
        vals = [s[k] for s in samples]
        m[k + "_ci"] = [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
    return m


def paired(rows1, rows2):
    a = {(r["scene"], r["instance_id"]): r for r in rows1}
    b = {(r["scene"], r["instance_id"]): r for r in rows2}
    keys = sorted(set(a) & set(b))
    lost = sum(a[k]["found50"] and not b[k]["found50"] for k in keys)
    gained = sum(not a[k]["found50"] and b[k]["found50"] for k in keys)
    n = lost + gained
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(lost, gained) + 1)) / 2 ** n)
    return {"parts": len(keys), "found_to_missed": lost, "missed_to_found": gained, "mcnemar_p": p,
            "mean_recall_change": float(np.mean([b[k]["recall"] - a[k]["recall"] for k in keys]))}


lines, result = [], {"t_mm": SETS}


def say(text=""):
    print(text)
    lines.append(text)


fmt = lambda m, k: f"{m[k]:.3f} [{m[k + '_ci'][0]:.3f}, {m[k + '_ci'][1]:.3f}]"
n_parts = len({(r["scene"], r["instance_id"]) for r in parts})
say(f"Evaluation: {n_parts} parts in {len({r['scene'] for r in scenes})} scenes. t: tuned on ideal "
    f"{t_ideal} mm (tied: {ties} mm), tuned on realistic {t_real} mm. 95 % CI over scenes.\n")

say("1. Conditions (B also with the tied values of t)")
result["conditions"] = {}
rows_spec = [("A", "tuned_on_ideal", "ideal"), ("B", "tuned_on_ideal", "realistic")]
rows_spec += [(f"B (t={t})", f"tuned_on_ideal_tie_t{t}", "realistic") for t in ties]
rows_spec += [("C", "tuned_on_realistic", "realistic")]
for cls in CLASSES:
    say(f"  {cls}")
    for name, params, variant in rows_spec:
        m = with_ci(select(parts, params, variant, cls), select(scenes, params, variant))
        result["conditions"][f"{name}|{cls}"] = m
        prec = f"  precision {fmt(m, 'precision')}  F1 {fmt(m, 'f1')}" if cls == "all" else ""
        say(f"    {name:9s} (t = {SETS[params]:2d} mm)  n={m['n_parts']:3d}  found50 {fmt(m, 'found50')}  "
            f"found75 {fmt(m, 'found75')}  recall {fmt(m, 'recall')}{prec}")

say("\n2. Paired comparisons (same parts): found (50 %) -> missed and back, mean recall change")
result["paired"] = {}
for x, y in (("A", "B"), ("B", "C"), ("A", "C")):
    for cls in CLASSES:
        p = paired(select(parts, *CONDITIONS[x], cls), select(parts, *CONDITIONS[y], cls))
        result["paired"][f"{x}->{y}|{cls}"] = p
        say(f"  {x} -> {y} {cls:8s}: found->missed {p['found_to_missed']:3d}, missed->found "
            f"{p['missed_to_found']:3d} (McNemar p = {p['mcnemar_p']:.2g}); mean recall change "
            f"{p['mean_recall_change']:+.3f}")

say("\n3. Ablation: t tuned on ideal, one camera effect at a time")
result["ablation"] = {}
for params in [k for k in SETS if k.startswith("tuned_on_ideal")]:
    say(f"  {params} (t = {SETS[params]} mm)")
    for variant in VARIANTS:
        m = with_ci(select(parts, params, variant), select(scenes, params, variant))
        result["ablation"][f"{params}|{variant}"] = m
        say(f"    {variant:10s} found50 {fmt(m, 'found50')}  precision {fmt(m, 'precision')}  "
            f"F1 {fmt(m, 'f1')}  recall {fmt(m, 'recall')}")

say("\n4. Fitted table plane vs true plane (median over scenes)")
for params in SETS:
    for variant in ("ideal", "realistic"):
        s = select(scenes, params, variant)
        say(f"  {params:28s} {variant:10s} angle {np.median([r['plane_angle_deg'] for r in s]):.3f} deg, "
            f"offset {np.median([r['plane_offset_mm'] for r in s]):+.2f} mm")

with open(os.path.join(args.exp, "analysis.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(args.exp, "analysis.json"), "w") as f:
    json.dump(result, f, indent=2)
tex = ["% generated by scripts/analyze_segment_experiment.py",
       "% condition & t [mm] & found50 & found75 & precision & F1 \\\\"]
for name, params, _ in rows_spec:
    m = result["conditions"][f"{name}|all"]
    tex.append(f"{name} & {SETS[params]} & {m['found50']:.3f} & {m['found75']:.3f} & {m['precision']:.3f} & "
               f"{m['f1']:.3f} \\\\")
with open(os.path.join(args.exp, "results_table.tex"), "w") as f:
    f.write("\n".join(tex) + "\n")

# ---------- figures ----------
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})


def save(fig, name):
    for folder in {args.exp, args.figdir}:
        fig.savefig(os.path.join(folder, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


# Threshold trade-off on the tuning scenes
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
t_grid = sorted({r["t_mm"] for r in tscenes})
for ax, (params, title) in zip(axes, (("tuned_on_ideal", "ideal depth"), ("tuned_on_realistic", "realistic depth"))):
    ms = [measures([r for r in tparts if r["params"] == params and r["t_mm"] == t],
                   [r for r in tscenes if r["params"] == params and r["t_mm"] == t]) for t in t_grid]
    for key, colour, ls, label in (("found50", BLUE, "-", "parts found (>= 50 % of pixels)"),
                                   ("precision", ORANGE, "-", "pixel precision"), ("f1", AQUA, "--", "F1")):
        ax.plot(t_grid, [m[key] for m in ms], color=colour, ls=ls, marker="o", ms=5, lw=2, label=label)
    ax.set_title(f"tuning scenes, {title}", color=INK)
    ax.set_xlabel("height threshold t above the table [mm]")
    ax.set_ylim(0, 1.03)
    ax.grid(color=GRID, lw=0.8)
axes[0].set_ylabel("share")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
save(fig, "fig_seg_threshold.png")


def dot_figure(entries, labels, name, title):
    """Rows = labels; left: parts found (50 %) per class; right: pixel precision (all parts)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(labels) + 1.5), sharey=True)
    y = np.arange(len(labels))[::-1]
    for k, (cls, colour, marker) in enumerate((("bracket", BLUE, "o"), ("profile", ORANGE, "s"))):
        ms = [entries(lab, cls) for lab in labels]
        x = np.array([m["found50"] for m in ms])
        err = np.array([[m["found50"] - m["found50_ci"][0], m["found50_ci"][1] - m["found50"]] for m in ms]).T
        axes[0].errorbar(x, y + (0.12 if k == 0 else -0.12), xerr=err, fmt=marker, color=colour, ms=7,
                         capsize=3, lw=1.5, label=f"{cls}s")
    ms = [entries(lab, "all") for lab in labels]
    x = np.array([m["precision"] for m in ms])
    err = np.array([[m["precision"] - m["precision_ci"][0], m["precision_ci"][1] - m["precision"]] for m in ms]).T
    axes[1].errorbar(x, y, xerr=err, fmt="D", color=INK2, ms=6, capsize=3, lw=1.5, label="all parts")
    axes[0].set_xlabel("share of parts found (>= 50 % of pixels)")
    axes[1].set_xlabel("pixel precision (pixels marked 'part' that are a part)")
    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis="x", color=GRID, lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, color=INK, y=1.07)
    save(fig, name)


cond_labels = [f"A  ideal -> ideal (t={t_ideal})", f"B  ideal -> realistic (t={t_ideal})"]
cond_labels += [f"B  ideal -> realistic (t={t}, tied)" for t in ties]
cond_labels += [f"C  realistic -> realistic (t={t_real})"]
cond_keys = dict(zip(cond_labels, [r[0] for r in rows_spec]))
dot_figure(lambda lab, cls: result["conditions"][f"{cond_keys[lab]}|{cls}"], cond_labels,
           "fig_seg_conditions.png", "Part/table segmentation in the three conditions (95 % CI over scenes)")
for cls in ("bracket", "profile"):       # per-class entries for the ablation figure
    for variant in VARIANTS:
        result["ablation"][f"tuned_on_ideal|{variant}|{cls}"] = with_ci(
            select(parts, "tuned_on_ideal", variant, cls), select(scenes, "tuned_on_ideal", variant))
abl_labels = ["ideal depth", "+ stereo shadows only", "+ edges only", "+ noise (matte level) only",
              "+ noise x2.9 (aluminium) only", "realistic (all effects)"]
abl_keys = dict(zip(abl_labels, ["ideal", "shadows", "edges", "noise", "aluminium", "realistic"]))
dot_figure(lambda lab, cls: result["ablation"][f"tuned_on_ideal|{abl_keys[lab]}" + ("" if cls == "all" else f"|{cls}")],
           abl_labels, "fig_seg_ablation.png",
           f"t tuned on ideal depth ({t_ideal} mm), one camera effect at a time (95 % CI over scenes)")
print(f"\nwritten in {args.exp}: analysis.txt, analysis.json, results_table.tex; figures also in "
      f"{args.figdir}: fig_seg_threshold.png, fig_seg_conditions.png, fig_seg_ablation.png")
