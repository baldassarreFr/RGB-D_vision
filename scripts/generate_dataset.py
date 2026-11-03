"""
Generate the synthetic dataset: N scenes with ideal RGB-D, labels and true poses.

Usage (from the project folder):
    python scripts/generate_dataset.py                 # 20 scenes, seeds 0..19
    python scripts/generate_dataset.py --n 50 --start-seed 100

Output:
    outputs/dataset/models/{profile,bracket}.ply    part models, in their own frame
    outputs/dataset/scenes/000000/ ...               one folder per scene (see scene.py)
    outputs/dataset/summary.json                     one line of facts per scene
    outputs/dataset/overview_rgb.png, overview_depth.png
"""
import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from rgbd_sim.scene import generate_scene, save_scene, save_models

MIN_VISIBLE_PIXELS = 100              # a part seen by fewer pixels is flagged
CROP = (slice(100, 400), slice(200, 650))   # image region around the drop area (rows, cols)

# Settings from the command line
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--n", type=int, default=20, help="number of scenes")
parser.add_argument("--start-seed", type=int, default=0, help="seed of the first scene")
parser.add_argument("--out", default="outputs/dataset", help="output folder")
args = parser.parse_args()

# Models, then one scene per seed
save_models(os.path.join(args.out, "models"))
summary, thumbnails = [], []
t_start = time.time()
for seed in range(args.start_seed, args.start_seed + args.n):
    scene = generate_scene(seed)
    save_scene(scene, os.path.join(args.out, "scenes", f"{seed:06d}"))

    meta = scene["meta"]
    classes = [o["class"] for o in meta["objects"]]
    visible = [o["visible_pixels"] for o in meta["objects"]]
    ok = meta["physics"]["settled"] and min(visible) >= MIN_VISIBLE_PIXELS
    summary.append({"seed": seed,
                    "profiles": classes.count("profile"),
                    "brackets": classes.count("bracket"),
                    "min_visible_pixels": min(visible),
                    "settled": meta["physics"]["settled"],
                    "ok": ok})
    thumbnails.append((seed, scene["rgb"][CROP], scene["depth"][CROP], classes))
    print(f"scene {seed:6d}: {classes.count('profile')} profile(s), {classes.count('bracket')} "
          f"bracket(s), min visible {min(visible):5d} px  {'ok' if ok else 'CHECK'}"
          f"   [{time.time() - t_start:.0f} s]")

with open(os.path.join(args.out, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
n_ok = sum(s["ok"] for s in summary)
print(f"done: {len(summary)} scenes, {n_ok} ok, in {time.time() - t_start:.0f} s -> {args.out}")


# Overview images saving
def overview(index, title, path, **imshow_args):
    cols = 5
    rows = int(np.ceil(len(thumbnails) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, item in zip(axes.ravel(), thumbnails):
        seed, image, classes = item[0], item[index], item[3]
        if index == 2:                                   # depth: draw empty pixels white
            image = np.where(image > 0, image, np.nan)
        ax.imshow(image, **imshow_args)
        ax.set_title(f"seed {seed}: {classes.count('profile')}P {classes.count('bracket')}B",
                     fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


overview(1, "RGB (P = profiles, B = brackets)", os.path.join(args.out, "overview_rgb.png"))
overview(2, "Ideal depth", os.path.join(args.out, "overview_depth.png"),
         cmap="viridis", interpolation="nearest")
