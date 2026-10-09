"""Classification heads over frozen image/text features.

Every model maps a batch dict to one logit per example (binary task) and is
built through `build_model(name, dims, **cfg)`. Batch keys:
    img, txt                 pooled embeddings (B, D_img), (B, D_txt)
    img_tok, txt_tok         token features (B, T, H)        - cross-attention only
    txt_mask                 (B, T_txt) bool, True = real token
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(d_in: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


def proj(d_in: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, d_out))


class Unimodal(nn.Module):
    """Baseline that sees only one modality."""
    uses_tokens = False

    def __init__(self, modality: str, dims: dict, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.key = {"image": "img", "text": "txt"}[modality]
        self.net = nn.Sequential(proj(dims[self.key], hidden, dropout), nn.GELU(), mlp(hidden, hidden, dropout))

    def forward(self, b):
        return self.net(b[self.key]).squeeze(-1)


class LateFusion(nn.Module):
    """Two unimodal heads whose logits are summed: no cross-modal interaction at all."""
    uses_tokens = False

    def __init__(self, dims: dict, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.img = Unimodal("image", dims, hidden, dropout)
        self.txt = Unimodal("text", dims, hidden, dropout)

    def forward(self, b):
        return self.img(b) + self.txt(b)


class ConcatFusion(nn.Module):
    """Early fusion: project each modality, concatenate, MLP."""
    uses_tokens = False

    def __init__(self, dims: dict, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.pi = proj(dims["img"], hidden, dropout)
        self.pt = proj(dims["txt"], hidden, dropout)
        self.head = mlp(2 * hidden, hidden, dropout)

    def forward(self, b):
        return self.head(F.gelu(torch.cat([self.pi(b["img"]), self.pt(b["txt"])], -1))).squeeze(-1)


class InteractionFusion(nn.Module):
    """Concat plus explicit multiplicative interactions [i, t, i*t, |i-t|].

    The elementwise product lets a shallow head express "this image AND this
    text" conjunctions that plain concatenation must learn through the MLP.
    """
    uses_tokens = False

    def __init__(self, dims: dict, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.pi = proj(dims["img"], hidden, dropout)
        self.pt = proj(dims["txt"], hidden, dropout)
        self.head = nn.Sequential(nn.LayerNorm(4 * hidden), mlp(4 * hidden, hidden, dropout))

    def forward(self, b):
        i, t = self.pi(b["img"]), self.pt(b["txt"])
        return self.head(torch.cat([i, t, i * t, (i - t).abs()], -1)).squeeze(-1)


class GatedFusion(nn.Module):
    """Gated Multimodal Unit (Arevalo et al., 2017).

    A per-dimension sigmoid gate z decides how much of the fused vector comes
    from the image vs. the text. `last_gate` keeps z for modality-reliance analysis.
    """
    uses_tokens = False

    def __init__(self, dims: dict, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.pi = proj(dims["img"], hidden, dropout)
        self.pt = proj(dims["txt"], hidden, dropout)
        self.gate = nn.Linear(2 * hidden, hidden)
        self.head = mlp(hidden, hidden, dropout)
        self.last_gate: torch.Tensor | None = None

    def forward(self, b):
        i, t = self.pi(b["img"]), self.pt(b["txt"])
        z = torch.sigmoid(self.gate(torch.cat([i, t], -1)))
        self.last_gate = z.detach()
        return self.head(z * torch.tanh(i) + (1 - z) * torch.tanh(t)).squeeze(-1)


class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention: text tokens query image patches and vice versa."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.t2i = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.i2t = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_t1, self.ln_i1 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ln_t2, self.ln_i2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff_t = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.ff_i = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.attn_t2i: torch.Tensor | None = None

    def forward(self, t, i, t_pad):
        # Pre-norm residual blocks. t_pad: True where the text position is padding.
        tn, i_n = self.ln_t1(t), self.ln_i1(i)
        dt, w = self.t2i(tn, i_n, i_n, need_weights=True, average_attn_weights=True)
        di, _ = self.i2t(i_n, tn, tn, key_padding_mask=t_pad, need_weights=False)
        self.attn_t2i = w.detach()  # (B, T_txt, T_img), kept for qualitative analysis
        t, i = t + dt, i + di
        t = t + self.ff_t(self.ln_t2(t))
        i = i + self.ff_i(self.ln_i2(i))
        return t, i


class CrossAttentionFusion(nn.Module):
    """Token-level fusion over cached encoder hidden states, plus the pooled embeddings."""
    uses_tokens = True

    def __init__(self, dims: dict, hidden: int = 256, dropout: float = 0.2, layers: int = 2, heads: int = 4):
        super().__init__()
        self.pi_tok = proj(dims["img_tok"], hidden, dropout)
        self.pt_tok = proj(dims["txt_tok"], hidden, dropout)
        self.blocks = nn.ModuleList(CrossAttentionBlock(hidden, heads, dropout) for _ in range(layers))
        self.pi = proj(dims["img"], hidden, dropout)
        self.pt = proj(dims["txt"], hidden, dropout)
        self.head = nn.Sequential(nn.LayerNorm(4 * hidden), mlp(4 * hidden, 2 * hidden, dropout))

    def forward(self, b):
        t, i = self.pt_tok(b["txt_tok"]), self.pi_tok(b["img_tok"])
        mask = b["txt_mask"]
        for blk in self.blocks:
            t, i = blk(t, i, ~mask)
        m = mask.unsqueeze(-1).to(t.dtype)
        t_pool = (t * m).sum(1) / m.sum(1).clamp(min=1.0)
        i_pool = i.mean(1)
        return self.head(torch.cat([t_pool, i_pool, self.pt(b["txt"]), self.pi(b["img"])], -1)).squeeze(-1)


MODELS = {
    "text_only": lambda dims, **kw: Unimodal("text", dims, **kw),
    "image_only": lambda dims, **kw: Unimodal("image", dims, **kw),
    "late": LateFusion,
    "concat": ConcatFusion,
    "interaction": InteractionFusion,
    "gated": GatedFusion,
    "cross_attn": CrossAttentionFusion,
}


def build_model(name: str, dims: dict, **kw) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](dims, **kw)
