"""Schematic of the pipeline and of the experiment (no data needed).

Usage: python scripts/make_design_figure.py      -> docs/figures/fig_design.png
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, AQUA, INK, INK2, FILL = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#f3f2ee"
plt.rcParams.update({"font.size": 9.5, "figure.facecolor": "white"})


def box(ax, x, y, w, h, title, body, edge=INK2, fill=FILL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=fill, ec=edge, lw=1.3))
    ax.text(x + w / 2, y + h - 0.025, title, ha="center", va="top", fontsize=10, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.075, body, ha="center", va="top", fontsize=8.5, color=INK2, linespacing=1.3)


def arrow(ax, p, q, colour=INK2, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=colour, lw=1.4, ls=ls))


fig, ax = plt.subplots(figsize=(11, 7.2))
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(0, 1)
ax.axis("off")

# ---------- top: the pipeline ----------
ax.text(0.0, 0.985, "1  Pipeline", fontsize=11, weight="bold", color=INK, va="top")
w, h, y = 0.215, 0.30, 0.62
steps = [("Synthetic scenes", "40x40 T-slot profiles and\nL-brackets (CAD), dropped\nwith PyBullet, ray-cast:\nideal depth, labels,\ntrue 6D poses"),
         ("D435i camera model", "from datasheet and papers\non the same camera:\nnoise (x2.9 on aluminium),\nedges moved by 1 px,\nstereo shadows"),
         ("Pose estimation", "part points (true labels)\n-> FPFH + RANSAC (coarse)\n-> ICP (refine)\n-> depth check over\n5 runs"),
         ("Error", "ADD-S (success if < 10 %\nof the part diameter),\nAUC of ADD-S,\ncentre and rotation\nerror")]
xs = [0.0, 0.26, 0.52, 0.785]
for x, (t, b) in zip(xs, steps):
    box(ax, x, y, w, h, t, b)
for x0, x1 in zip(xs[:-1], xs[1:]):
    arrow(ax, (x0 + w + 0.005, y + h / 2), (x1 - 0.005, y + h / 2))
ax.text(0.26 + w / 2, y - 0.02, "6 variants: ideal, noise, aluminium,\nshadows, edges, realistic (all)",
        ha="center", va="top", fontsize=8, color=INK2)

# ---------- bottom: the experiment ----------
ax.text(0.0, 0.52, "2  Experiment: same parts, same pipeline; only the depth images differ",
        fontsize=11, weight="bold", color=INK, va="top")
box(ax, 0.0, 0.07, 0.19, 0.37, "Tuning scenes", "seeds 0-19 (93 parts)\n\nchoose voxel size v\nand tolerance d\n(grid search, per class,\nhighest ADD-S AUC)")
box(ax, 0.25, 0.29, 0.19, 0.15, "tuned on ideal depth", "-> parameters P_ideal", edge=BLUE)
box(ax, 0.25, 0.07, 0.19, 0.15, "tuned on realistic depth", "-> parameters P_realistic", edge=AQUA)
arrow(ax, (0.195, 0.33), (0.245, 0.365))
arrow(ax, (0.195, 0.18), (0.245, 0.145))
box(ax, 0.50, 0.07, 0.19, 0.37, "Evaluation scenes", "seeds 20-119 (420 parts)\nused once, never for\nchoosing parameters")
conds = [("A", "P_ideal on ideal depth", "what ideal synthetic data promise", BLUE, 0.375),
         ("B", "P_ideal on realistic depth", "same pipeline, realistic images", ORANGE, 0.255),
         ("C", "P_realistic on realistic depth", "camera modelled in development", AQUA, 0.135)]
for name, what, meaning, colour, yy in conds:
    ax.add_patch(FancyBboxPatch((0.75, yy - 0.045), 0.25, 0.09, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc="white", ec=colour, lw=1.8))
    ax.text(0.762, yy + 0.022, f"{name}", fontsize=12, weight="bold", color=INK, va="top")
    ax.text(0.79, yy + 0.025, what, fontsize=8.5, color=INK, va="top")
    ax.text(0.79, yy - 0.008, meaning, fontsize=8, color=INK2, va="top")
    arrow(ax, (0.695, 0.255), (0.745, yy), colour=colour)
arrow(ax, (0.445, 0.365), (0.495, 0.32), colour=BLUE)
arrow(ax, (0.445, 0.145), (0.495, 0.19), colour=AQUA)
ax.text(0.5, 0.0, "Answers:   A \u2212 B = how much ideal data overstate the result;   "
        "C \u2212 B = what modelling the camera buys;   A \u2212 C = what remains",
        ha="center", fontsize=9, color=INK)

os.makedirs("docs/figures", exist_ok=True)
fig.savefig("docs/figures/fig_design.png", dpi=200, bbox_inches="tight")
print("written docs/figures/fig_design.png")
