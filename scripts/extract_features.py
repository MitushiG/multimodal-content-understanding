"""Cache frozen-encoder features for every split.

    python scripts/extract_features.py --data data/hateful_memes --encoders clip-b32 bert dinov2-s \
        --tokens clip-b32
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmcu.data import resolve_splits  # noqa: E402
from mmcu.encoders import Encoder, pick_device  # noqa: E402
from mmcu.features import cache_features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/hateful_memes")
ap.add_argument("--out", default="features")
ap.add_argument("--encoders", nargs="+", default=["clip-b32", "bert", "dinov2-s"])
ap.add_argument("--tokens", nargs="*", default=["clip-b32"], help="encoders whose token states are cached too")
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--device", default=None)
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()

splits = resolve_splits(args.data)
print({k: len(v) for k, v in splits.items()})
device = pick_device(args.device)
for name in args.encoders:
    enc = Encoder(name, device)
    cache_features(enc, splits, args.out, args.batch_size, tokens=name in args.tokens, overwrite=args.overwrite)
    del enc
