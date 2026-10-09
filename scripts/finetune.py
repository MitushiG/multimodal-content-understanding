"""End-to-end fine-tuning of CLIP + interaction-fusion head (run on a GPU, e.g. Colab T4).

The last `--unfreeze` transformer blocks of *both* towers are trained (with a
small LR) together with the fusion head (larger LR); everything below stays frozen.
`--unfreeze 0` reproduces the frozen-feature setting end-to-end as a sanity check.

    python scripts/finetune.py --data data/hateful_memes --encoder clip-b16 --unfreeze 2 --seed 0

Writes results/runs/<name>/seed<k>.{npz,json} in the same format as
run_experiments.py, so scripts/analyze.py picks it up automatically.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmcu.data import load_image, resolve_splits  # noqa: E402
from mmcu.encoders import REGISTRY, pick_device  # noqa: E402
from mmcu.fusion import InteractionFusion  # noqa: E402
from mmcu.metrics import best_threshold, evaluate  # noqa: E402
from mmcu.train import set_seed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/hateful_memes")
ap.add_argument("--encoder", default="clip-b32", choices=[k for k, v in REGISTRY.items() if v[1] == "clip"])
ap.add_argument("--unfreeze", type=int, default=2, help="number of top transformer blocks to train per tower")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--grad-accum", type=int, default=1)
ap.add_argument("--lr-backbone", type=float, default=1e-5)
ap.add_argument("--lr-head", type=float, default=5e-4)
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--dropout", type=float, default=0.2)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--device", default=None)
ap.add_argument("--out", default="results/runs")
ap.add_argument("--max-train", type=int, default=None, help="subsample train (smoke tests)")
args = ap.parse_args()

set_seed(args.seed)
device = pick_device(args.device)
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

hf_id = REGISTRY[args.encoder][0]
processor = CLIPProcessor.from_pretrained(hf_id)
splits = resolve_splits(args.data)
if args.max_train:
    splits["train"] = splits["train"][: args.max_train]


class MemeDataset(Dataset):
    def __init__(self, exs):
        self.exs = exs

    def __len__(self):
        return len(self.exs)

    def __getitem__(self, i):
        e = self.exs[i]
        return load_image(e.img_path), e.text, e.label


def collate(batch):
    imgs, txts, ys = zip(*batch)
    enc = processor(text=list(txts), images=list(imgs), return_tensors="pt", padding="max_length",
                    truncation=True, max_length=77)
    return enc["pixel_values"], enc["input_ids"], enc["attention_mask"], torch.tensor(ys, dtype=torch.float32)


class CLIPFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(hf_id)
        d = self.clip.config.projection_dim
        self.head = InteractionFusion({"img": d, "txt": d}, hidden=args.hidden, dropout=args.dropout)

    def forward(self, px, ids, mask):
        v = self.clip.vision_model(pixel_values=px).pooler_output
        t = self.clip.text_model(input_ids=ids, attention_mask=mask).pooler_output
        return self.head({"img": self.clip.visual_projection(v), "txt": self.clip.text_projection(t)})


model = CLIPFusion().to(device)
for p in model.clip.parameters():
    p.requires_grad = False
backbone = []
if args.unfreeze > 0:
    vm, tm = model.clip.vision_model, model.clip.text_model
    trainable = (list(vm.encoder.layers[-args.unfreeze:]) + [vm.post_layernorm, model.clip.visual_projection]
                 + list(tm.encoder.layers[-args.unfreeze:]) + [tm.final_layer_norm, model.clip.text_projection])
    for m in trainable:
        for p in m.parameters():
            p.requires_grad = True
            backbone.append(p)
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {n_train / 1e6:.2f}M (backbone {sum(p.numel() for p in backbone) / 1e6:.2f}M)")

groups = [{"params": list(model.head.parameters()), "lr": args.lr_head}]
if backbone:
    groups.append({"params": backbone, "lr": args.lr_backbone})
opt = torch.optim.AdamW(groups, weight_decay=1e-2)
loaders = {s: DataLoader(MemeDataset(v), batch_size=args.batch_size, shuffle=(s == "train"),
                         num_workers=args.workers, collate_fn=collate) for s, v in splits.items()}
steps = args.epochs * len(loaders["train"]) // args.grad_accum
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=max(1, steps),
                                            pct_start=0.1)
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
loss_fn = nn.BCEWithLogitsLoss()


@torch.no_grad()
def predict(split):
    model.eval()
    out = []
    for px, ids, mask, _ in loaders[split]:
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out.append(torch.sigmoid(model(px.to(device), ids.to(device), mask.to(device)).float()).cpu())
    return torch.cat(out).numpy()


from sklearn.metrics import roc_auc_score  # noqa: E402

y = {s: np.array([e.label for e in v]) for s, v in splits.items()}
best, best_epoch, best_state, history = -1.0, 0, None, []
for epoch in range(args.epochs):
    model.train()
    t0, total = time.time(), 0.0
    for step, (px, ids, mask, yb) in enumerate(loaders["train"]):
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(model(px.to(device), ids.to(device), mask.to(device)).float(), yb.to(device))
        scaler.scale(loss / args.grad_accum).backward()
        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            if sched.last_epoch < steps:
                sched.step()
        total += loss.item() * len(yb)
    val = roc_auc_score(y["val"], predict("val"))
    history.append({"epoch": epoch, "train_loss": total / len(splits["train"]), "val_auroc": float(val),
                    "sec": time.time() - t0})
    print(history[-1])
    if val > best:
        best, best_epoch = val, epoch
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    elif epoch - best_epoch >= args.patience:
        break

model.load_state_dict(best_state)
name = f"{args.encoder.replace('-', '')}_finetune_top{args.unfreeze}"
out_dir = Path(args.out) / name
out_dir.mkdir(parents=True, exist_ok=True)
probs = {s: predict(s) for s in splits if s != "train"}
thr = best_threshold(y["val"], probs["val"])
metrics = {s: evaluate(y[s], p, thr) for s, p in probs.items()}
np.savez_compressed(out_dir / f"seed{args.seed}.npz", **probs)
exp = {"name": name, "image": args.encoder, "text": args.encoder, "model": "finetune", "group": "rq4",
       "unfreeze": args.unfreeze}
json.dump({"config": vars(args), "history": history, "best_epoch": best_epoch, "best_val_auroc": best,
           "experiment": exp, "threshold": thr, "metrics": metrics, "trainable_params": n_train},
          open(out_dir / f"seed{args.seed}.json", "w"), indent=1)
print({s: round(m["auroc"], 4) for s, m in metrics.items()})
