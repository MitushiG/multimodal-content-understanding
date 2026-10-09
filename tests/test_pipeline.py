"""Fast unit tests (no model downloads): shapes, metrics, slices, training on toy features."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmcu.analysis import confounder_slices, pairwise_accuracy  # noqa: E402
from mmcu.fusion import MODELS, build_model  # noqa: E402
from mmcu.metrics import evaluate  # noqa: E402
from mmcu.train import TrainConfig, predict, train  # noqa: E402

DIMS = {"img": 12, "txt": 10, "img_tok": 8, "txt_tok": 6}


def toy_features(n=400, seed=0, tokens=True):
    """XOR task: label = sign(img[0]) != sign(txt[0]); each modality alone is uninformative."""
    g = torch.Generator().manual_seed(seed)
    img, txt = torch.randn(n, DIMS["img"], generator=g), torch.randn(n, DIMS["txt"], generator=g)
    y = ((img[:, 0] > 0) != (txt[:, 0] > 0)).long()
    f = {"ids": [str(i) for i in range(n)], "label": y, "image": {"emb": img}, "text": {"emb": txt}}
    if tokens:
        f["image"]["tokens"] = torch.randn(n, 5, DIMS["img_tok"], generator=g)
        f["text"]["tokens"] = torch.randn(n, 4, DIMS["txt_tok"], generator=g)
        f["text"]["mask"] = torch.ones(n, 4, dtype=torch.bool)
        f["text"]["mask"][:, 3] = False
    return f


@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_shapes(name):
    m = build_model(name, DIMS, hidden=16, dropout=0.0)
    f = toy_features(8)
    b = {"img": f["image"]["emb"], "txt": f["text"]["emb"], "img_tok": f["image"]["tokens"],
         "txt_tok": f["text"]["tokens"], "txt_mask": f["text"]["mask"]}
    assert m(b).shape == (8,)


def test_fusion_learns_xor_and_unimodal_cannot():
    tr, va = toy_features(1500, 0, tokens=False), toy_features(400, 1, tokens=False)
    ys = va["label"].numpy()
    res = {}
    for name in ("text_only", "late", "interaction"):
        m, _ = train(TrainConfig(model=name, hidden=64, max_epochs=40, patience=10, lr=3e-3), tr, va, log=lambda *_: None)
        res[name] = evaluate(ys, predict(m, va, torch.device("cpu")))["auroc"]
    assert res["text_only"] < 0.62 and res["late"] < 0.62
    assert res["interaction"] > 0.85


def test_metrics_and_pairs():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = evaluate(y, p, threshold=0.5)
    assert m["auroc"] == pytest.approx(0.75) and m["confusion"] == [[2, 0], [1, 1]]
    assert pairwise_accuracy(np.array([0.9, 0.2, 0.5, 0.5]), [(0, 1), (2, 3)]) == pytest.approx(0.75)


def test_confounder_slices():
    texts = ["Same words!", "same words", "other"]
    emb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y = np.array([1, 0, 1])
    ids = ["a", "b", "c"]
    s = confounder_slices(ids, texts, y, emb, ids, texts, y, emb, 0.95)
    assert s["text_ambiguous"].tolist() == [True, True, False]   # caption repeats with both labels
    assert s["image_ambiguous"].tolist() == [False, True, True]  # b and c share an image, differ in label
    assert s["text_pairs"] == [(0, 1)] and s["image_pairs"] == [(2, 1)]
