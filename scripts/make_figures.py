"""Figures that show the procedure on real examples (for the report and the README).

Usage (after the experiment):  python scripts/make_figures.py
Reads outputs/dataset_eval, outputs/experiment/evaluation.csv and chosen.json; writes to docs/figures:
  fig_scene.png        one evaluation scene: colour image, labels, ideal depth, realistic depth
  fig_pose_bracket.png pose estimation of a typical bracket, on ideal depth (A) and on the
  fig_pose_profile.png realistic depth (B): measured depth, pose after RANSAC, final pose;
                       outline of the model in the estimated pose (orange) and in the true pose (blue)
  fig_error_axes.png   where the error of condition B lies: along the camera's line of sight or
                       sideways (re-estimates a sample of parts; prints the medians)
"Typical" = the part whose centre error in B is closest to the median of its class.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # same settings as the experiment

import argparse
import csv
import json
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba

from rgbd_sim.pose import estimate_pose_verified, model_point_cloud
from rgbd_sim.render import render_ideal
from rgbd_sim.scene import MAKE_PART, load_depth, load_scene

BLUE, ORANGE, AQUA, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": "white"})

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--eval", default="outputs/dataset_eval")
parser.add_argument("--exp", default="outputs/experiment")
parser.add_argument("--figdir", default="docs/figures")
parser.add_argument("--sample", type=int, default=30, help="parts per class for fig_error_axes")
parser.add_argument("--workers", type=int, default=os.cpu_count())
args = parser.parse_args()
os.makedirs(args.figdir, exist_ok=True)
with open(os.path.join(args.exp, "chosen.json")) as f:
    chosen = json.load(f)
with open(os.path.join(args.exp, "evaluation.csv"), newline="") as f:
    rows = [r for r in csv.DictReader(f)]
for r in rows:
    r["centre_mm"] = float(r["centre_mm"])
    r["instance_id"] = int(r["instance_id"])

_parts = {}


def _init():
    for cls, make in MAKE_PART.items():
        mesh = make()
        _parts[cls] = {"mesh": mesh, "model": model_point_cloud(mesh)}


def estimate(job):
    """Pose of one part with the parameters tuned on ideal depth (conditions A and B)."""
    scene_name, inst, cls, variant = job
    folder = os.path.join(args.eval, "scenes", scene_name)
    scene, depth = load_scene(folder), load_depth(folder, variant)
    p = chosen["tuned_on_ideal"][cls]
    pose, info = estimate_pose_verified(depth, scene["label"], inst, np.array(scene["meta"]["camera"]["K"]),
                                        _parts[cls]["mesh"], _parts[cls]["model"], p["voxel_mm"] / 1000,
                                        tolerance=p["tolerance_mm"] / 1000, seed=0)
    return pose, info


def save(fig, name):
    fig.savefig(os.path.join(args.figdir, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("written", os.path.join(args.figdir, name))


def crop_of(mask, margin):
    r, c = np.where(mask)
    return (slice(max(r.min() - margin, 0), r.max() + margin + 1),
            slice(max(c.min() - margin, 0), c.max() + margin + 1))


def show_depth(ax, depth, crop, vmin, vmax):
    img = np.where(depth > 0, depth, np.nan)[crop] * 1000
    cmap = plt.get_cmap("Greys_r").copy()
    cmap.set_bad("white")
    ax.imshow(img, cmap=cmap, vmin=vmin * 1000, vmax=vmax * 1000, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def outline(mesh, pose, K, shape):
    """Pixels on the border of the model's silhouette when placed with `pose`."""
    placed = mesh.copy()
    placed.apply_transform(pose)
    d, _ = render_ideal([placed], K, np.eye(4), shape[1], shape[0])
    m = d > 0
    inner = m.copy()
    inner[1:, :] &= m[:-1, :]
    inner[:-1, :] &= m[1:, :]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    return m & ~inner


def draw_outline(ax, edge, crop, colour):
    rgba = np.zeros(edge[crop].shape + (4,))
    rgba[edge[crop]] = to_rgba(colour)
    ax.imshow(rgba, interpolation="nearest")


if __name__ == "__main__":
    # ---------- 1. One evaluation scene ----------
    scene_dirs = sorted(os.listdir(os.path.join(args.eval, "scenes")))
    name = next(s for s in scene_dirs
                if {o["class"] for o in load_scene(os.path.join(args.eval, "scenes", s))["meta"]["objects"]}
                == {"profile", "bracket"})
    folder = os.path.join(args.eval, "scenes", name)
    scene = load_scene(folder)
    realistic = load_depth(folder, "realistic")
    crop = crop_of(scene["label"] >= 2, 25)
    vmin, vmax = scene["depth"][crop][scene["depth"][crop] > 0].min(), scene["depth"][crop].max()
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.2))
    ax[0, 0].imshow(scene["rgb"])
    ax[0, 0].set_title("(a) colour image (Blinn-Phong shading)")
    n_inst = int(scene["label"].max()) + 1
    colours = ["white", "#d9d8d4"] + [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#6250d6",
                                      "#e34948"][: max(n_inst - 2, 0)]
    ax[0, 1].imshow(scene["label"][crop], cmap=ListedColormap(colours[:n_inst]), vmin=0, vmax=n_inst - 1,
                    interpolation="nearest")
    ax[0, 1].set_title("(b) labels: one colour per part (table grey)")
    show_depth(ax[1, 0], scene["depth"], crop, vmin, vmax)
    ax[1, 0].set_title("(c) ideal depth")
    show_depth(ax[1, 1], realistic, crop, vmin, vmax)
    ax[1, 1].set_title("(d) realistic depth (D435i model): noise, edges, stereo shadows (white)")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"Evaluation scene {name} (depth: darker = closer)", color=INK)
    plt.tight_layout()
    save(fig, "fig_scene.png")

    # ---------- 2. Pose estimation of a typical part, A and B ----------
    for cls in ("bracket", "profile"):
        b = [r for r in rows if r["params"] == "tuned_on_ideal" and r["variant"] == "realistic" and r["cls"] == cls]
        med = np.nanmedian([r["centre_mm"] for r in b])
        pick = min(b, key=lambda r: abs(r["centre_mm"] - med))
        folder = os.path.join(args.eval, "scenes", pick["scene"])
        scene = load_scene(folder)
        K = np.array(scene["meta"]["camera"]["K"])
        obj = next(o for o in scene["meta"]["objects"] if o["instance_id"] == pick["instance_id"])
        true = np.array(obj["pose_camera"])
        _init()
        mesh = _parts[cls]["mesh"]
        crop = crop_of(scene["label"] == pick["instance_id"], 12)
        edge_true = outline(mesh, true, K, scene["depth"].shape)
        fig, ax = plt.subplots(2, 3, figsize=(10, 6.5))
        for i, (variant, cond) in enumerate((("ideal", "A: ideal depth"), ("realistic", "B: realistic depth"))):
            depth = load_depth(folder, variant)
            pose, info = estimate((pick["scene"], pick["instance_id"], cls, variant))
            vals = depth[crop][depth[crop] > 0]
            for j, (title, est) in enumerate((("measured depth of the part", None),
                                              ("after FPFH + RANSAC", info["coarse_pose"]),
                                              ("final (ICP + depth check)", pose))):
                a = ax[i, j]
                show_depth(a, depth, crop, vals.min(), vals.max())
                if est is not None:
                    draw_outline(a, edge_true, crop, BLUE)
                    draw_outline(a, outline(mesh, est, K, depth.shape), crop, ORANGE)
                    err = np.linalg.norm(est[:3, 3] - true[:3, 3]) * 1000
                    title += f"\ncentre error {err:.1f} mm"
                a.set_title(title, fontsize=9)
            ax[i, 0].set_ylabel(cond, fontsize=10)
        fig.suptitle(f"A typical {cls} (scene {pick['scene']}): model outline in the estimated pose "
                     f"(orange) and in the true pose (blue); 1 pixel = about 1.5 mm", color=INK, fontsize=10)
        plt.tight_layout()
        save(fig, f"fig_pose_{cls}.png")

    # ---------- 3. Where is the error of B: along the line of sight or sideways? ----------
    rng = np.random.default_rng(0)
    jobs, truth = [], []
    for cls in ("bracket", "profile"):
        b = [r for r in rows if r["params"] == "tuned_on_ideal" and r["variant"] == "realistic" and r["cls"] == cls]
        for k in rng.choice(len(b), min(args.sample, len(b)), replace=False):
            r = b[k]
            jobs.append((r["scene"], r["instance_id"], cls, "realistic"))
            truth.append(r)
    with Pool(args.workers, initializer=_init) as pool:
        results = pool.map(estimate, jobs)
    along, side, cls_of, check = [], [], [], []
    for (scene_name, inst, cls, _), (pose, _), r in zip(jobs, results, truth):
        meta = load_scene(os.path.join(args.eval, "scenes", scene_name))["meta"]
        true = np.array(next(o for o in meta["objects"] if o["instance_id"] == inst)["pose_camera"])
        e = (pose[:3, 3] - true[:3, 3]) * 1000
        u = true[:3, 3] / np.linalg.norm(true[:3, 3])          # line of sight to the part's centre
        along.append(abs(e @ u))
        side.append(np.linalg.norm(e - (e @ u) * u))
        cls_of.append(cls)
        check.append(abs(np.linalg.norm(e) - r["centre_mm"]))
    along, side, cls_of = np.array(along), np.array(side), np.array(cls_of)
    print(f"re-estimated {len(jobs)} parts; largest difference from the experiment's centre error: "
          f"{max(check):.3f} mm (0 = same poses as in the experiment)")
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = max(along.max(), side.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls=":")
    ax.text(lim * 0.97, lim * 0.9, "equal", color=INK2, ha="right", fontsize=8)
    for cls, colour, marker in (("bracket", BLUE, "o"), ("profile", ORANGE, "s")):
        m = cls_of == cls
        ax.scatter(side[m], along[m], color=colour, marker=marker, s=28, alpha=0.8,
                   label=f"{cls}s: median {np.median(side[m]):.1f} mm sideways, "
                         f"{np.median(along[m]):.1f} mm along")
        print(f"{cls}: median sideways {np.median(side[m]):.2f} mm, along the line of sight "
              f"{np.median(along[m]):.2f} mm (n = {m.sum()})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("centre error sideways [mm]")
    ax.set_ylabel("centre error along the line of sight [mm]")
    ax.set_title("Condition B: direction of the centre error", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(color="#e4e3df", lw=0.8)
    save(fig, "fig_error_axes.png")
