"""Train every experiment x seed on cached features; save predictions and metrics.

    python scripts/run_experiments.py --config configs/experiments.yaml [--only clip_concat ...]

Writes results/runs/<experiment>/seed<k>.npz (probabilities for every eval split,
plus reliance probes on test) and seed<k>.json (config, history, metrics).
Existing runs are skipped, so the script is safe to re-run after an interruption.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmcu.features import load_pair  # noqa: E402
from mmcu.metrics import best_threshold, evaluate  # noqa: E402
from mmcu.train import TrainConfig, predict, train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/experiments.yaml")
ap.add_argument("--features", default="features")
ap.add_argument("--out", default="results/runs")
ap.add_argument("--only", nargs="*", help="experiment names to run (default: all)")
ap.add_argument("--seeds", nargs="*", type=int, help="override seeds from the config")
ap.add_argument("--device", default=None, help="default: cpu for pooled heads, mps/cuda for cross_attn")
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
seeds = args.seeds if args.seeds is not None else cfg["seeds"]
exps = [e for e in cfg["experiments"] if not args.only or e["name"] in args.only]
TRAIN_KEYS = set(TrainConfig.__dataclass_fields__)


def available_splits(enc: str) -> list[str]:
    return sorted(p.stem for p in (Path(args.features) / enc).glob("*.pt"))


def default_device(model: str) -> str:
    if args.device:
        return args.device
    if model != "cross_attn":
        return "cpu"  # small MLPs: CPU beats MPS kernel-launch overhead
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


cache: dict = {}


def feats(img: str, txt: str, split: str) -> dict:
    key = (img, txt, split)
    if key not in cache:
        cache[key] = load_pair(args.features, img, txt, split)
    return cache[key]


for exp in exps:
    eval_splits = [s for s in available_splits(exp["image"]) if s != "train"]
    out_dir = Path(args.out) / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        stem = out_dir / f"seed{seed}"
        if stem.with_suffix(".json").exists() and not args.overwrite:
            print(f"skip {stem}")
            continue
        merged = {**cfg["defaults"], **exp, "seed": seed, "device": default_device(exp["model"])}
        tc = TrainConfig(**{k: v for k, v in merged.items() if k in TRAIN_KEYS})
        tr = feats(exp["image"], exp["text"], "train")
        va = feats(exp["image"], exp["text"], "val")
        model, info = train(tc, tr, va)
        dev = torch.device(tc.device)

        probs, metrics = {}, {}
        p_val = predict(model, va, dev)
        thr = best_threshold(va["label"].numpy(), p_val)
        for split in eval_splits:
            f = feats(exp["image"], exp["text"], split)
            p = p_val if split == "val" else predict(model, f, dev)
            probs[split] = p
            metrics[split] = evaluate(f["label"].numpy(), p, thr)
        # Reliance probes: how much does test AUROC drop when a modality is broken?
        if "test" in eval_splits and exp["model"] not in ("text_only", "image_only"):
            te = feats(exp["image"], exp["text"], "test")
            for probe in ("shuffle_image", "shuffle_text", "zero_image", "zero_text"):
                p = predict(model, te, dev, perturb=probe, seed=seed)
                probs[f"test__{probe}"] = p
                metrics[f"test__{probe}"] = evaluate(te["label"].numpy(), p, thr)

        np.savez_compressed(stem.with_suffix(".npz"), **probs)
        json.dump({**info, "experiment": exp, "threshold": thr, "metrics": metrics},
                  open(stem.with_suffix(".json"), "w"), indent=1)
        print(f"  {exp['name']} seed {seed}: " + "  ".join(
            f"{s}={m['auroc']:.3f}" for s, m in metrics.items() if "__" not in s))
