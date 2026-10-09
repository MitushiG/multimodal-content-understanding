# When does multimodal fusion help? Image + text content understanding on Hateful Memes

A reproducible study of vision-language fusion for policy-violating content detection.
Each image and caption can be harmless on its own while the pair is violating, the
same situation a trust & safety system faces with ad creatives.

**Stack:** PyTorch · Hugging Face Transformers · CLIP (ViT) · BERT · DINOv2 · scikit-learn

## Research questions

| | Question | How it is tested |
|---|---|---|
| **RQ1** | Does fusing image + text beat either modality alone, and which fusion architecture works best? | Text-only and image-only baselines vs. late fusion, concat, multiplicative interaction, gated (GMU), and token-level cross-attention heads on frozen CLIP features. 5 seeds, bootstrap CIs, paired bootstrap tests vs. the best unimodal model (chosen on validation). |
| **RQ2** | Does it matter that the encoders were pretrained to be *aligned*? | Same heads on CLIP (aligned image–text space) vs. BERT + DINOv2 (independently pretrained). |
| **RQ3** | Does the model actually use both modalities, and where does fusion fail? | Reliance probes (shuffle/zero one modality at test time), confounder slices, modality-dropout training, error analysis. |
| **RQ4** | What does end-to-end fine-tuning add over frozen features? | Unfreeze the top *k* transformer blocks of both CLIP towers (k = 0, 2, 4) on a Colab T4, plus latency measurements. |

### The confounder analysis (the core of RQ3)

Hateful Memes was built with *benign confounders*: the same caption on a different
image, or the same image with a different caption, flips the label. `src/mmcu/analysis.py`
recovers these structurally:

- **same-caption pairs**: identical (normalized) caption, opposite labels → only the image can decide.
- **same-image pairs**: near-duplicate image (CLIP cosine ≥ 0.95), opposite labels → only the text can decide.

**Pair accuracy** is the probability that a model scores the hateful member of such a pair
higher. A text-only model scores exactly 0.5 on same-caption pairs by construction, so this
metric isolates genuine cross-modal reasoning from aggregate AUROC.

## Results

*Fill in from `results/report.md` after running.* The headline table, significance
tests, reliance table and slice table are all generated there, along with
`fig_auroc.png`, `fig_confounders.png`, `fig_reliance.png` and `fig_confusion.png`.

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests          # 10 fast unit tests, no downloads
```

**Data.** Hateful Memes is licensed by Meta: accept the agreement at
<https://hatefulmemeschallenge.com/> and download it. Do not redistribute the data in this
repo. Unzip it so that `data/hateful_memes/` contains `img/` and the `*.jsonl` files. Both
the Phase-1 layout (`train/dev/test`) and the Phase-2 layout (`dev_seen`, `test_seen`, `*_unseen`)
work. `data.resolve_splits` always produces a labeled train / val / test, carving val out of
train if necessary.

```bash
# 1. cache frozen features (M1 laptop, MPS); token states only for CLIP (~2 GB)
python scripts/extract_features.py --data data/hateful_memes --encoders clip-b32 bert dinov2-s --tokens clip-b32
# 2. RQ1-RQ3: every head x 5 seeds on cached features (minutes, CPU)
python scripts/run_experiments.py --config configs/experiments.yaml
# 3. RQ4 on Colab: notebooks/colab_finetune.ipynb, then copy its runs into results/runs/
# 4. latency on this machine
python scripts/latency.py --encoder clip-b32 --batch-sizes 1 32
# 5. tables, tests, slices, figures -> results/
python scripts/analyze.py --data data/hateful_memes
```

**Smoke test without the real data:** `scripts/make_synthetic.py` builds a toy dataset where
the label is "caption contradicts image", so each modality alone carries zero signal.
A correct pipeline shows unimodal and late-fusion AUROC ≈ 0.5 and interaction-capable
fusion ≈ 1.0, which this repo reproduces:

```bash
python scripts/make_synthetic.py --out data/synthetic
python scripts/extract_features.py --data data/synthetic --out features_synth
python scripts/run_experiments.py --features features_synth --out results_synth/runs --seeds 0 1
python scripts/analyze.py --data data/synthetic --features features_synth --runs results_synth/runs --out results_synth
```

## Layout

```
src/mmcu/
  data.py        Hateful Memes loading + split resolution
  encoders.py    frozen CLIP / BERT / DINOv2 wrappers (pooled + token features)
  features.py    feature caching; joins image/text features from different encoders
  fusion.py      text_only, image_only, late, concat, interaction, gated (GMU), cross_attn
  train.py       training loop, early stopping on val AUROC, modality dropout, reliance probes
  metrics.py     AUROC, macro-F1, per-class P/R, confusion, ECE, bootstrap CIs, paired tests
  analysis.py    confounder slices, pair accuracy, error tables
scripts/         extract_features, run_experiments, finetune, latency, analyze, make_synthetic
configs/experiments.yaml   every experiment in the study
notebooks/colab_finetune.ipynb
```

## Design decisions and caveats

- **Frozen features first.** Caching encoder outputs makes 5-seed sweeps over 13 configurations
  run in minutes on a laptop; end-to-end fine-tuning is isolated to RQ4.
- **Model selection on validation only.** Early stopping, the decision threshold (max macro-F1)
  and the "best unimodal baseline" used in significance tests are all chosen on val, never on test.
- **CLIP center-crops to a square**, which can cut off text at the edges of wide memes. The
  OCR'd caption is still provided as text, but some visual typography is lost.
- **Confounder detection is heuristic.** Caption matching is exact after normalization, and
  image matching uses a cosine threshold (`--sim-thresh`). Spot-check a few pairs before
  quoting slice numbers.
- Hateful Memes is a research proxy for ad policy enforcement, not a substitute. Label taxonomy,
  base rates and adversarial pressure differ in production.
