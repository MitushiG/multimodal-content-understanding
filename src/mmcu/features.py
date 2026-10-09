"""Cache frozen-encoder features to disk so every fusion experiment trains in seconds.

Layout: <feat_root>/<encoder>/<split>.pt holding
    {"ids": [str], "label": LongTensor(N),
     "image": {"emb": (N,D), "tokens": (N,T,H) fp16}?,   # present if encoder has that tower
     "text":  {"emb": (N,D), "tokens": (N,T,H) fp16, "mask": (N,T)}?}
Tokens are only stored with `tokens=True` (they are ~1 GB per split set for CLIP-B/32).
"""
from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from .data import Example, load_image
from .encoders import Encoder


def _batches(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def extract_split(enc: Encoder, exs: list[Example], batch_size: int = 32, tokens: bool = False) -> dict:
    out: dict = {"ids": [e.id for e in exs], "label": torch.tensor([e.label for e in exs])}
    for mod in enc.modalities:
        embs, toks, masks = [], [], []
        for chunk in tqdm(list(_batches(exs, batch_size)), desc=f"{enc.name}:{mod}", leave=False):
            if mod == "image":
                r = enc.encode_images([load_image(e.img_path) for e in chunk])
            else:
                r = enc.encode_texts([e.text for e in chunk])
            embs.append(r["emb"])
            if tokens:
                toks.append(r["tokens"].half())
                if "mask" in r:
                    masks.append(r["mask"])
        d = {"emb": torch.cat(embs)}
        if tokens:
            d["tokens"] = torch.cat(toks)
            if masks:
                d["mask"] = torch.cat(masks)
                # Trim padding columns no example uses: memes are short, CLIP pads to 77.
                t_max = int(d["mask"].sum(1).max())
                d["tokens"], d["mask"] = d["tokens"][:, :t_max], d["mask"][:, :t_max]
        out[mod] = d
    return out


def cache_features(enc: Encoder, splits: dict[str, list[Example]], feat_root: str | Path,
                   batch_size: int = 32, tokens: bool = False, overwrite: bool = False) -> None:
    d = Path(feat_root) / enc.name
    d.mkdir(parents=True, exist_ok=True)
    for split, exs in splits.items():
        path = d / f"{split}.pt"
        if path.exists() and not overwrite:
            print(f"skip {path} (exists)")
            continue
        torch.save(extract_split(enc, exs, batch_size, tokens), path)
        print(f"wrote {path}  n={len(exs)}")


def load_pair(feat_root: str | Path, image_enc: str, text_enc: str, split: str) -> dict:
    """Join image features from one encoder with text features from another (possibly the same)."""
    root = Path(feat_root)
    fi = torch.load(root / image_enc / f"{split}.pt", weights_only=False)
    ft = fi if text_enc == image_enc else torch.load(root / text_enc / f"{split}.pt", weights_only=False)
    if fi["ids"] != ft["ids"]:
        raise ValueError(f"id order mismatch between {image_enc} and {text_enc} on {split}")
    if "image" not in fi or "text" not in ft:
        raise ValueError(f"{image_enc} must provide images and {text_enc} texts")
    return {"ids": fi["ids"], "label": fi["label"], "image": fi["image"], "text": ft["text"]}
