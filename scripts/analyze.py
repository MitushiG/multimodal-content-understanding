"""Aggregate runs into tables, significance tests, confounder-slice analysis and figures.

    python scripts/analyze.py --data data/hateful_memes

Outputs (results/):
    report.md              all tables, ready to paste into a write-up
    summary.json           the same numbers, machine-readable
    errors_<model>.csv     most confident mistakes of the best fusion model
    fig_*.png              figures
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmcu.analysis import confounder_slices, error_table, pairwise_accuracy, slice_metrics  # noqa: E402
from mmcu.data import resolve_splits  # noqa: E402
from mmcu.metrics import best_threshold, bootstrap_ci, evaluate, paired_bootstrap_delta  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/hateful_memes")
ap.add_argument("--features", default="features")
ap.add_argument("--runs", default="results/runs")
ap.add_argument("--out", default="results")
ap.add_argument("--slice-encoder", default="clip-b32", help="image embeddings used to find near-duplicate images")
ap.add_argument("--sim-thresh", type=float, default=0.95)
args = ap.parse_args()
out = Path(args.out)

# Palette (validated categorical order; see README "Figures").
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

# ---------------------------------------------------------------- load runs
runs = {}
for d in sorted(Path(args.runs).iterdir()):
    seeds = sorted(d.glob("seed*.json"))
    if not seeds:
        continue
    infos = [json.load(open(s)) for s in seeds]
    probs = [dict(np.load(s.with_suffix(".npz"))) for s in seeds]
    runs[d.name] = {"infos": infos, "probs": probs, "exp": infos[0]["experiment"]}
if not runs:
    sys.exit(f"no runs under {args.runs}")

splits = resolve_splits(args.data)
eval_splits = [s for s in runs[next(iter(runs))]["probs"][0] if "__" not in s and s != "val"]
labels = {s: np.array([e.label for e in splits[s]]) for s in splits}
texts = {s: [e.text for e in splits[s]] for s in splits}
ids = {s: [e.id for e in splits[s]] for s in splits}


def ens(name, split):
    """Seed-ensemble probability (mean over seeds)."""
    return np.mean([p[split] for p in runs[name]["probs"]], axis=0)


summary = {"models": {}}
lines = ["# Results\n"]

# ---------------------------------------------------------------- RQ tables
lines.append("## Main results (mean ± std over seeds; ensemble = seed-averaged probabilities)\n")
for split in eval_splits:
    lines.append(f"### {split} (n={len(labels[split])})\n")
    lines.append("| group | model | AUROC | acc | macro-F1 | ECE | ensemble AUROC [95% CI] |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, r in runs.items():
        ms = [i["metrics"][split] for i in r["infos"]]
        agg = {k: (np.mean([m[k] for m in ms]), np.std([m[k] for m in ms]))
               for k in ("auroc", "accuracy", "macro_f1", "ece")}
        y, p = labels[split], ens(name, split)
        thr = best_threshold(labels["val"], ens(name, "val"))
        em = evaluate(y, p, thr)
        ci = bootstrap_ci(y, p)
        summary["models"].setdefault(name, {})[split] = {
            "seeds": {k: {"mean": v[0], "std": v[1]} for k, v in agg.items()},
            "ensemble": em, "ensemble_auroc_ci": ci}
        lines.append(f"| {r['exp'].get('group', '')} | {name} | " + " | ".join(
            f"{agg[k][0]:.3f} ± {agg[k][1]:.3f}" for k in ("auroc", "accuracy", "macro_f1", "ece"))
            + f" | {em['auroc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |")
    lines.append("")

test = "test" if "test" in eval_splits else eval_splits[0]
y_test = labels[test]
val_auroc = {n: np.mean([i["best_val_auroc"] for i in r["infos"]]) for n, r in runs.items()}
unimodal = [n for n, r in runs.items() if r["exp"]["model"] in ("text_only", "image_only")]
fusion = [n for n in runs if n not in unimodal]

# ---------------------------------------------------------------- significance
if unimodal and fusion:
    lines.append(f"## Is fusion significantly better than its best unimodal baseline? ({test}, paired bootstrap)\n")
    lines.append("| fusion model | vs | ΔAUROC | 95% CI | p (Δ≤0) |")
    lines.append("|---|---|---|---|---|")
    summary["significance"] = {}
    for f in fusion:
        enc = (runs[f]["exp"]["image"], runs[f]["exp"]["text"])
        same = [u for u in unimodal if (runs[u]["exp"]["image"], runs[u]["exp"]["text"]) == enc] or unimodal
        base = max(same, key=val_auroc.get)  # chosen on validation, not test
        d = paired_bootstrap_delta(y_test, ens(f, test), ens(base, test))
        summary["significance"][f] = {"baseline": base, **d}
        lines.append(f"| {f} | {base} | {d['delta']:+.3f} | [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | {d['p_value']:.3f} |")
    lines.append("")

# ---------------------------------------------------------------- modality reliance
probe_rows = []
lines.append(f"## Modality reliance ({test}): AUROC when one modality is broken\n")
lines.append("| model | intact | shuffle image | shuffle text | zero image | zero text |")
lines.append("|---|---|---|---|---|---|")
for f in fusion:
    if f"{test}__shuffle_image" not in runs[f]["probs"][0]:
        continue
    vals = {k: np.mean([i["metrics"][f"{test}__{k}"]["auroc"] if k != "intact" else i["metrics"][test]["auroc"]
                        for i in runs[f]["infos"]])
            for k in ("intact", "shuffle_image", "shuffle_text", "zero_image", "zero_text")}
    probe_rows.append((f, vals))
    summary["models"][f]["reliance"] = vals
    lines.append(f"| {f} | " + " | ".join(f"{vals[k]:.3f}" for k in vals) + " |")
lines.append("\nShuffling keeps each modality's marginal distribution but breaks the pairing; "
             "a model that truly fuses should drop toward 0.5 under *both* shuffles.\n")

# ---------------------------------------------------------------- confounder slices
slice_feats = Path(args.features) / args.slice_encoder
if (slice_feats / f"{test}.pt").exists():
    emb = {s: torch.load(slice_feats / f"{s}.pt", weights_only=False)["image"]["emb"] for s in splits}
    all_ids = sum((ids[s] for s in splits), [])
    sl = confounder_slices(ids[test], texts[test], y_test, emb[test],
                           all_ids, sum((texts[s] for s in splits), []),
                           np.concatenate([labels[s] for s in splits]),
                           torch.cat([emb[s] for s in splits]), args.sim_thresh)
    masks = {"all": np.ones(len(y_test), bool), "text_ambiguous": sl["text_ambiguous"],
             "image_ambiguous": sl["image_ambiguous"]}
    t_name = max([u for u in unimodal if runs[u]["exp"]["model"] == "text_only"], key=val_auroc.get, default=None)
    i_name = max([u for u in unimodal if runs[u]["exp"]["model"] == "image_only"], key=val_auroc.get, default=None)
    if t_name and i_name:
        pt = ens(t_name, test) >= best_threshold(labels["val"], ens(t_name, "val"))
        pi = ens(i_name, test) >= best_threshold(labels["val"], ens(i_name, "val"))
        masks["unimodal_disagree"] = pt != pi
    rest = ~(sl["text_ambiguous"] | sl["image_ambiguous"])
    masks["unambiguous"] = rest

    lines.append(f"## When does fusion help? Confounder slices ({test})\n")
    lines.append(f"Slice sizes: " + ", ".join(f"{k}={int(m.sum())}" for k, m in masks.items())
                 + f"; within-split confounder pairs: caption={len(sl['text_pairs'])}, image={len(sl['image_pairs'])}.\n")
    lines.append("| model | " + " | ".join(f"acc: {k}" for k in masks) + " | pair-acc: same caption | pair-acc: same image |")
    lines.append("|---|" + "---|" * (len(masks) + 2))
    summary["slices"] = {"sizes": {k: int(m.sum()) for k, m in masks.items()},
                         "n_text_pairs": len(sl["text_pairs"]), "n_image_pairs": len(sl["image_pairs"]),
                         "models": {}}
    for name in runs:
        p = ens(name, test)
        thr = best_threshold(labels["val"], ens(name, "val"))
        sm = {k: slice_metrics(y_test, p, m, thr) for k, m in masks.items()}
        pa_t, pa_i = pairwise_accuracy(p, sl["text_pairs"]), pairwise_accuracy(p, sl["image_pairs"])
        summary["slices"]["models"][name] = {**sm, "pair_acc_text": pa_t, "pair_acc_image": pa_i}
        lines.append(f"| {name} | " + " | ".join(f"{sm[k]['accuracy']:.3f}" for k in masks)
                     + f" | {pa_t:.3f} | {pa_i:.3f} |")
    lines.append("\nPair accuracy = P(model scores the hateful member of a confounder pair higher); 0.5 = chance. "
                 "A text-only model is exactly 0.5 on same-caption pairs by construction.\n")
else:
    sl = None
    print(f"(skipping slice analysis: no {slice_feats}/{test}.pt)")

# ---------------------------------------------------------------- errors
best_fusion = max(fusion, key=val_auroc.get) if fusion else max(runs, key=val_auroc.get)
thr = best_threshold(labels["val"], ens(best_fusion, "val"))
err = error_table(ids[test], texts[test], y_test, {n: ens(n, test) for n in runs}, best_fusion, thr)
err.to_csv(out / f"errors_{best_fusion}.csv", index=False)
lines.append(f"## Error analysis\n\nBest fusion model on validation: **{best_fusion}**. "
             f"Its most confident mistakes are in `errors_{best_fusion}.csv`, with every model's score per row.\n")
summary["best_fusion"] = best_fusion

# ---------------------------------------------------------------- figures
def legend_above(ax, items):
    ax.legend(handles=[Patch(color=c, label=l) for c, l in items], frameon=False, ncol=len(items),
              loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.0, borderaxespad=0.2)


order = sorted(runs, key=lambda n: summary["models"][n][test]["ensemble"]["auroc"])
fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(order) + 0.9))
for k, n in enumerate(order):
    s = summary["models"][n][test]
    lo, hi = s["ensemble_auroc_ci"]
    col = BLUE if n in fusion else ORANGE
    ax.plot([lo, hi], [k, k], color=col, lw=2, solid_capstyle="round")
    ax.plot(s["ensemble"]["auroc"], k, "o", ms=8, color=col, mec="white", mew=2)
ax.set_yticks(range(len(order)), order)
ax.set_xlabel(f"{test} AUROC (seed ensemble, 95% bootstrap CI)")
ax.grid(axis="y", visible=False)
legend_above(ax, [(BLUE, "fusion"), (ORANGE, "unimodal")])
fig.savefig(out / "fig_auroc.png")
plt.close(fig)

if probe_rows:
    probe_rows.sort(key=lambda r: order.index(r[0]))
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(probe_rows) + 1.0))
    h = 0.36
    for k, (n, v) in enumerate(probe_rows):
        ax.barh(k + h / 2, v["intact"] - v["shuffle_image"], h * 0.9, color=BLUE)
        ax.barh(k - h / 2, v["intact"] - v["shuffle_text"], h * 0.9, color=ORANGE)
    ax.set_yticks(range(len(probe_rows)), [n for n, _ in probe_rows])
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel(f"AUROC drop on {test} when the modality is shuffled across examples")
    legend_above(ax, [(BLUE, "image shuffled"), (ORANGE, "text shuffled")])
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "fig_reliance.png")
    plt.close(fig)

if sl is not None and (sl["text_pairs"] or sl["image_pairs"]):
    names = order
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(names) + 1.0))
    h = 0.36
    for k, n in enumerate(names):
        s = summary["slices"]["models"][n]
        ax.barh(k + h / 2, np.nan_to_num(s["pair_acc_text"]), h * 0.9, color=BLUE)
        ax.barh(k - h / 2, np.nan_to_num(s["pair_acc_image"]), h * 0.9, color=AQUA)
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("confounder pair accuracy (0.5 = chance)")
    legend_above(ax, [(BLUE, f"same caption, n={len(sl['text_pairs'])}"),
                      (AQUA, f"same image, n={len(sl['image_pairs'])}")])
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "fig_confounders.png")
    plt.close(fig)

cm = np.array(summary["models"][best_fusion][test]["ensemble"]["confusion"])
fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#cde2fb", "#104281"]))
for (r, c), v in np.ndenumerate(cm):
    ax.text(c, r, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else INK)
ax.set_xticks([0, 1], ["benign", "hateful"])
ax.set_yticks([0, 1], ["benign", "hateful"])
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.grid(False)
ax.set_title(f"{best_fusion} ({test})", fontsize=10, color=INK)
fig.savefig(out / "fig_confusion.png")
plt.close(fig)

(out / "report.md").write_text("\n".join(lines))
json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
print("\n".join(lines))
