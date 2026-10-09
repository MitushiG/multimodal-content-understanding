"""Hateful Memes data loading.

Expected layout (the official Phase-2 release, also mirrored on Kaggle/HF):

    data/hateful_memes/
        img/01235.png ...
        train.jsonl            {"id": 1235, "img": "img/01235.png", "label": 0, "text": "..."}
        dev_seen.jsonl         labeled, 500 examples
        dev_unseen.jsonl       labeled, 540 examples  (optional)
        test_seen.jsonl        labeled in Phase-2, 1000 examples (optional)
        test_unseen.jsonl      labeled in Phase-2, 2000 examples (optional)

Older releases ship `dev.jsonl` / `test.jsonl` instead; `test.jsonl` there is
unlabeled. `resolve_splits` maps whatever exists onto train / val / test so the
rest of the pipeline never needs to know which release you downloaded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

LABEL_NAMES = ["benign", "hateful"]


@dataclass
class Example:
    id: str
    img_path: Path
    text: str
    label: int  # -1 when unlabeled


def read_jsonl(path: Path, root: Path) -> list[Example]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append(Example(
                id=str(r["id"]),
                img_path=root / r["img"],
                text=r.get("text", ""),
                label=int(r["label"]) if "label" in r else -1,
            ))
    return out


def _labeled(exs: list[Example]) -> bool:
    return len(exs) > 0 and all(e.label in (0, 1) for e in exs)


def resolve_splits(root: str | Path, val_frac: float = 0.1, seed: int = 0) -> dict[str, list[Example]]:
    """Return {"train", "val", "test", ...extra labeled eval splits} from whatever files exist.

    Priority:
      val  = dev_seen  > dev  > held-out slice of train
      test = test_seen > (dev if dev was not used as val) > none
      dev_unseen / test_unseen are kept as extra evaluation splits when labeled.
    """
    root = Path(root)
    files = {p.stem: read_jsonl(p, root) for p in sorted(root.glob("*.jsonl"))}
    if "train" not in files:
        raise FileNotFoundError(f"No train.jsonl under {root}")

    train = files["train"]
    splits: dict[str, list[Example]] = {}

    if _labeled(files.get("dev_seen", [])):
        val = files["dev_seen"]
    elif _labeled(files.get("dev", [])):
        val = files["dev"]
    else:
        val = None

    if _labeled(files.get("test_seen", [])):
        test = files["test_seen"]
    elif val is not None and val is not files.get("dev") and _labeled(files.get("dev", [])):
        test = files["dev"]
    else:
        test = None

    # Fallbacks keep the pipeline runnable on partial releases: carve labeled
    # splits out of train, stratified by label.
    rng = np.random.default_rng(seed)
    if test is None and val is not None:
        test, val = val, None  # prefer a genuinely held-out test; rebuild val from train
    if val is None:
        y = np.array([e.label for e in train])
        val_idx = []
        for c in (0, 1):
            idx = np.flatnonzero(y == c)
            rng.shuffle(idx)
            val_idx.extend(idx[: max(1, int(round(val_frac * len(idx))))])
        val_set = set(val_idx)
        val = [train[i] for i in sorted(val_set)]
        train = [e for i, e in enumerate(train) if i not in val_set]

    splits["train"], splits["val"] = train, val
    if test is not None:
        splits["test"] = test
    for extra in ("dev_unseen", "test_unseen"):
        if _labeled(files.get(extra, [])):
            splits[extra] = files[extra]
    return splits


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def labels(exs: list[Example]) -> np.ndarray:
    return np.array([e.label for e in exs], dtype=np.int64)
