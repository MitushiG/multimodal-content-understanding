"""Slices that test *when* multimodal fusion should matter.

Hateful Memes contains "benign confounders": the same caption on a different
image (or the same image with a different caption) flips the label. On those
examples one modality is provably insufficient, which is exactly the
Ads Trust situation of "harmless image + harmless text = violating ad".

  text_ambiguous   the example's caption appears elsewhere in the dataset with the
                   opposite label  ->  the image has to decide
  image_ambiguous  a near-duplicate image (CLIP cosine >= threshold) appears elsewhere
                   with the opposite label  ->  the text has to decide
  unimodal_disagree  the text-only and image-only baselines predict different classes

Pairwise confounder accuracy: for pairs *inside the evaluation split* that share a
caption (or image) but differ in label, how often does the model score the
hateful member higher? A text-only model is stuck at exactly 0.5 on caption pairs.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t.lower())).strip()


def _near_dup_neighbors(q: torch.Tensor, ref: torch.Tensor, thresh: float, chunk: int = 1024):
    """For each query row, indices of reference rows with cosine >= thresh."""
    q = torch.nn.functional.normalize(q.float(), dim=1)
    ref = torch.nn.functional.normalize(ref.float(), dim=1)
    out = []
    for s in range(0, len(q), chunk):
        sim = q[s:s + chunk] @ ref.T
        out.extend([torch.nonzero(r >= thresh).flatten().tolist() for r in sim])
    return out


def confounder_slices(eval_ids, eval_texts, eval_labels, eval_img_emb,
                      all_ids, all_texts, all_labels, all_img_emb, sim_thresh: float = 0.95) -> dict:
    """Boolean masks over the eval split + within-split confounder pairs."""
    eval_labels, all_labels = np.asarray(eval_labels), np.asarray(all_labels)
    by_text = defaultdict(set)
    for t, y in zip(all_texts, all_labels):
        by_text[normalize_text(t)].add(int(y))
    text_amb = np.array([len(by_text[normalize_text(t)]) == 2 for t in eval_texts])

    nbrs = _near_dup_neighbors(eval_img_emb, all_img_emb, sim_thresh)
    img_amb = np.zeros(len(eval_ids), dtype=bool)
    for k, (eid, y, nb) in enumerate(zip(eval_ids, eval_labels, nbrs)):
        img_amb[k] = any(all_ids[j] != eid and all_labels[j] != y for j in nb)

    # Within-split pairs (indices into the eval split).
    text_pairs, image_pairs = [], []
    groups = defaultdict(list)
    for k, t in enumerate(eval_texts):
        groups[normalize_text(t)].append(k)
    for ks in groups.values():
        text_pairs += [(a, b) for a in ks for b in ks if eval_labels[a] == 1 and eval_labels[b] == 0]
    self_nbrs = _near_dup_neighbors(eval_img_emb, eval_img_emb, sim_thresh)
    for a, nb in enumerate(self_nbrs):
        image_pairs += [(a, b) for b in nb if eval_labels[a] == 1 and eval_labels[b] == 0]

    return {"text_ambiguous": text_amb, "image_ambiguous": img_amb,
            "text_pairs": text_pairs, "image_pairs": image_pairs}


def pairwise_accuracy(p: np.ndarray, pairs: list[tuple[int, int]], tol: float = 1e-5) -> float:
    """Fraction of (hateful, benign) pairs ranked correctly; ties (within float noise) count 1/2."""
    if not pairs:
        return float("nan")
    a = np.array([p[i] for i, _ in pairs])
    b = np.array([p[j] for _, j in pairs])
    tie = np.abs(a - b) <= tol
    return float(((a > b) & ~tie).mean() + 0.5 * tie.mean())


def slice_metrics(y: np.ndarray, p: np.ndarray, mask: np.ndarray, threshold: float) -> dict:
    m = np.asarray(mask, dtype=bool)
    if m.sum() == 0:
        return {"n": 0, "accuracy": float("nan"), "auroc": float("nan")}
    ys, ps = y[m], p[m]
    return {
        "n": int(m.sum()),
        "accuracy": float(((ps >= threshold) == ys).mean()),
        "auroc": float(roc_auc_score(ys, ps)) if len(np.unique(ys)) == 2 else float("nan"),
    }


def error_table(ids, texts, y, probs: dict[str, np.ndarray], focus: str, threshold: float, k: int = 25):
    """Most confident false positives / negatives of model `focus`, with every model's score alongside."""
    import pandas as pd
    df = pd.DataFrame({"id": ids, "text": texts, "label": y, **{f"p_{m}": v for m, v in probs.items()}})
    p = df[f"p_{focus}"]
    fp = df[(p >= threshold) & (df.label == 0)].sort_values(f"p_{focus}", ascending=False).head(k)
    fn = df[(p < threshold) & (df.label == 1)].sort_values(f"p_{focus}").head(k)
    return pd.concat([fp.assign(error="false_positive"), fn.assign(error="false_negative")])
