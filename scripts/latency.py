"""Inference latency: frozen encoder forward + fusion head, from preprocessed tensors.

    python scripts/latency.py --encoder clip-b32 --batch-sizes 1 32

Reports median / p95 milliseconds per batch and per example. Excludes disk I/O and
image decoding, includes tokenization/preprocessing (it is part of serving cost).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmcu.encoders import Encoder, pick_device  # noqa: E402
from mmcu.fusion import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--encoder", default="clip-b32")
ap.add_argument("--head", default="interaction")
ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32])
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--device", default=None)
ap.add_argument("--out", default="results/latency.json")
args = ap.parse_args()

device = pick_device(args.device)
enc = Encoder(args.encoder, device)


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


rng = np.random.default_rng(0)
img = Image.fromarray(rng.integers(0, 255, (400, 400, 3), dtype=np.uint8))
probe = enc.encode_images([img]), enc.encode_texts(["warm up"])
dims = {"img": probe[0]["emb"].shape[1], "txt": probe[1]["emb"].shape[1]}
head = build_model(args.head, dims).to(device).eval()

results = []
for bs in args.batch_sizes:
    imgs, txts = [img] * bs, ["when you see the price of gas this week"] * bs
    times = []
    for it in range(args.iters + 5):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            i, t = enc.encode_images(imgs), enc.encode_texts(txts)
            head({"img": i["emb"].to(device), "txt": t["emb"].to(device)})
        sync()
        if it >= 5:  # warm-up
            times.append((time.perf_counter() - t0) * 1000)
    r = {"encoder": args.encoder, "head": args.head, "device": str(device), "batch_size": bs,
         "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
         "per_example_ms": float(np.median(times) / bs)}
    results.append(r)
    print(r)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(args.out, "w"), indent=1)
