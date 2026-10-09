"""Frozen pretrained encoders that turn (image, text) into feature tensors.

Each encoder exposes `encode_images` / `encode_texts`, returning a dict with
  "emb"    (B, D)       pooled embedding used by the pooled fusion heads
  "tokens" (B, T, H)    last-layer hidden states, for cross-attention fusion
  "mask"   (B, T) bool  valid-token mask (text only; image patches are all valid)

Registry names are what configs and the CLI refer to. CLIP gives *aligned*
image/text spaces; BERT + DINOv2 are an *unaligned* pair of independently
pretrained unimodal encoders, which is one of the ablations.
"""
from __future__ import annotations

import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

REGISTRY = {
    # name: (hf_id, kind, modalities)
    "clip-b32": ("openai/clip-vit-base-patch32", "clip", ("image", "text")),
    "clip-b16": ("openai/clip-vit-base-patch16", "clip", ("image", "text")),
    "clip-l14": ("openai/clip-vit-large-patch14", "clip", ("image", "text")),
    "bert": ("bert-base-uncased", "text", ("text",)),
    "dinov2-s": ("facebook/dinov2-small", "image", ("image",)),
    "dinov2-b": ("facebook/dinov2-base", "image", ("image",)),
}


def pick_device(pref: str | None = None) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Encoder:
    def __init__(self, name: str, device: torch.device, text_max_len: int = 128):
        if name not in REGISTRY:
            raise KeyError(f"Unknown encoder {name!r}; choose from {sorted(REGISTRY)}")
        self.name = name
        self.hf_id, self.kind, self.modalities = REGISTRY[name]
        self.device = device
        if self.kind == "clip":
            self.model = CLIPModel.from_pretrained(self.hf_id).to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(self.hf_id)
            self.text_max_len = 77
        elif self.kind == "text":
            self.model = AutoModel.from_pretrained(self.hf_id).to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
            self.text_max_len = text_max_len
        else:
            self.model = AutoModel.from_pretrained(self.hf_id).to(device).eval()
            self.processor = AutoImageProcessor.from_pretrained(self.hf_id)

    @torch.no_grad()
    def encode_images(self, images) -> dict[str, torch.Tensor]:
        if "image" not in self.modalities:
            raise ValueError(f"{self.name} has no image tower")
        if self.kind == "clip":
            px = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
            out = self.model.vision_model(pixel_values=px)
            emb = self.model.visual_projection(out.pooler_output)
        else:
            px = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
            out = self.model(pixel_values=px)
            emb = out.pooler_output  # layer-normed CLS for DINOv2
        return {"emb": emb.float().cpu(), "tokens": out.last_hidden_state.float().cpu()}

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:
        if "text" not in self.modalities:
            raise ValueError(f"{self.name} has no text tower")
        tok = self.processor.tokenizer if self.kind == "clip" else self.tokenizer
        enc = tok(texts, padding="max_length", truncation=True, max_length=self.text_max_len,
                  return_tensors="pt").to(self.device)
        if self.kind == "clip":
            out = self.model.text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            emb = self.model.text_projection(out.pooler_output)
        else:
            out = self.model(**enc)
            m = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1.0)  # mean-pool
        return {
            "emb": emb.float().cpu(),
            "tokens": out.last_hidden_state.float().cpu(),
            "mask": enc["attention_mask"].bool().cpu(),
        }
