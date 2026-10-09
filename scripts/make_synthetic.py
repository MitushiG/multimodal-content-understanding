"""Tiny synthetic image+text dataset in Hateful Memes format, for smoke tests.

Label = 1 iff the caption *contradicts* the image (red image + "grass" caption, or
green image + "fire" caption). Each modality alone carries zero label information,
so a correct pipeline must show unimodal AUROC ~0.5 and fusion AUROC >> 0.5.

    python scripts/make_synthetic.py --out data/synthetic
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic")
ap.add_argument("--n", nargs=3, type=int, default=[600, 150, 200], metavar=("TRAIN", "DEV", "TEST"))
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rng = random.Random(args.seed)
root = Path(args.out)
(root / "img").mkdir(parents=True, exist_ok=True)
TEMPLATES = {"fire": ["look at this fire", "the fire is spreading", "so much fire today"],
             "grass": ["look at this grass", "the grass is growing", "so much grass today"]}
COLORS = {"red": (200, 40, 30), "green": (40, 160, 50)}

k = 0
for split, n in zip(["train", "dev_seen", "test_seen"], args.n):
    with open(root / f"{split}.jsonl", "w") as f:
        for _ in range(n):
            color, word = rng.choice(list(COLORS)), rng.choice(list(TEMPLATES))
            base = tuple(max(0, min(255, c + rng.randint(-25, 25))) for c in COLORS[color])
            im = Image.new("RGB", (160, 160), base)
            d = ImageDraw.Draw(im)
            for _ in range(rng.randint(1, 4)):
                x, y, s = rng.randint(0, 130), rng.randint(0, 130), rng.randint(10, 30)
                d.ellipse([x, y, x + s, y + s], fill=tuple(rng.randint(0, 255) for _ in range(3)))
            path = f"img/{k:05d}.png"
            im.save(root / path)
            label = int((color == "red") != (word == "fire"))
            f.write(json.dumps({"id": k, "img": path, "label": label, "text": rng.choice(TEMPLATES[word])}) + "\n")
            k += 1
print(f"wrote {k} examples to {root}")
