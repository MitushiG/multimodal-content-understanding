"""Train fusion heads on cached features, with early stopping on validation AUROC."""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .fusion import build_model


@dataclass
class TrainConfig:
    model: str = "concat"
    hidden: int = 512
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    max_epochs: int = 40
    patience: int = 6
    modality_dropout: float = 0.0   # per-example prob. of zeroing each modality during training
    pos_weight: bool = False        # reweight the positive class by n_neg / n_pos
    seed: int = 0
    device: str = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_dims(f: dict) -> dict:
    d = {"img": f["image"]["emb"].shape[1], "txt": f["text"]["emb"].shape[1]}
    if "tokens" in f["image"]:
        d["img_tok"] = f["image"]["tokens"].shape[2]
    if "tokens" in f["text"]:
        d["txt_tok"] = f["text"]["tokens"].shape[2]
    return d


def make_batch(f: dict, idx, device, use_tokens: bool, perturb: str | None = None) -> dict:
    """Slice a batch; `perturb` = zero_image / zero_text blanks one modality (reliance probe)."""
    b = {"img": f["image"]["emb"][idx].float(), "txt": f["text"]["emb"][idx].float()}
    if use_tokens:
        b["img_tok"] = f["image"]["tokens"][idx].float()
        b["txt_tok"] = f["text"]["tokens"][idx].float()
        b["txt_mask"] = f["text"]["mask"][idx]
    if perturb == "zero_image":
        b["img"] = torch.zeros_like(b["img"])
        if use_tokens:
            b["img_tok"] = torch.zeros_like(b["img_tok"])
    elif perturb == "zero_text":
        b["txt"] = torch.zeros_like(b["txt"])
        if use_tokens:
            b["txt_tok"] = torch.zeros_like(b["txt_tok"])
    return {k: v.to(device) for k, v in b.items()}


def _apply_modality_dropout(b: dict, p: float) -> None:
    """Zero image or text per example (never both), so the model can't lean on one modality."""
    n = b["img"].shape[0]
    drop_img = torch.rand(n, device=b["img"].device) < p
    drop_txt = (torch.rand(n, device=b["img"].device) < p) & ~drop_img
    for key, m in (("img", drop_img), ("img_tok", drop_img), ("txt", drop_txt), ("txt_tok", drop_txt)):
        if key in b:
            shape = (n,) + (1,) * (b[key].dim() - 1)
            b[key] = b[key] * (~m).view(shape).to(b[key].dtype)


@torch.no_grad()
def predict(model: nn.Module, f: dict, device, batch_size: int = 256, perturb: str | None = None,
            seed: int = 0) -> np.ndarray:
    """P(hateful) for every example. With `perturb="shuffle_*"` the permutation spans the whole split."""
    model.eval()
    n = len(f["label"])
    use_tok = getattr(model, "uses_tokens", False)
    rng = np.random.default_rng(seed)
    if perturb and perturb.startswith("shuffle"):
        # Permute globally once, then batch in order, so shuffled partners come from the whole split.
        perm = rng.permutation(n)
        key = "image" if perturb == "shuffle_image" else "text"
        f = {**f, key: {k: v[perm] for k, v in f[key].items()}}
        perturb = None
    out = []
    for s in range(0, n, batch_size):
        idx = np.arange(s, min(s + batch_size, n))
        out.append(torch.sigmoid(model(make_batch(f, idx, device, use_tok, perturb))).cpu())
    return torch.cat(out).numpy()


def train(cfg: TrainConfig, train_f: dict, val_f: dict, log=print) -> tuple[nn.Module, dict]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    kw = {"hidden": cfg.hidden, "dropout": cfg.dropout}
    model = build_model(cfg.model, feature_dims(train_f), **kw).to(device)
    use_tok = getattr(model, "uses_tokens", False)
    if use_tok and "tokens" not in train_f["image"]:
        raise ValueError("cross_attn needs token features: re-run extract_features.py with --tokens")

    y = train_f["label"].numpy()
    pw = torch.tensor((y == 0).sum() / max(1, (y == 1).sum())) if cfg.pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw.to(device) if pw is not None else None)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)

    rng = np.random.default_rng(cfg.seed)
    y_val = val_f["label"].numpy()
    best = {"auroc": -1.0, "epoch": -1, "state": None}
    history = []
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = rng.permutation(len(y))
        total = 0.0
        for s in range(0, len(perm), cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            b = make_batch(train_f, idx, device, use_tok)
            if cfg.modality_dropout > 0:
                _apply_modality_dropout(b, cfg.modality_dropout)
            target = torch.as_tensor(y[idx], dtype=torch.float32, device=device)
            loss = loss_fn(model(b), target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        val_auroc = roc_auc_score(y_val, predict(model, val_f, device))
        history.append({"epoch": epoch, "train_loss": total / len(y), "val_auroc": float(val_auroc)})
        if val_auroc > best["auroc"]:
            best = {"auroc": float(val_auroc), "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= cfg.patience:
            break
    model.load_state_dict(best["state"])
    log(f"[{cfg.model} seed={cfg.seed}] best val AUROC {best['auroc']:.4f} @ epoch {best['epoch']}")
    return model, {"config": asdict(cfg), "history": history, "best_epoch": best["epoch"],
                   "best_val_auroc": best["auroc"]}
