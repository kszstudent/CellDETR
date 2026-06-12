# CellDETR: A Detection-Guided Framework for Scalable Cell Representation Learning from Histopathology Images

This repository contains the official PyTorch implementation of **CellDETR**, a detection-guided framework built on Deformable DETR for scalable cell representation learning from whole-slide images (WSIs).

> **CellDETR: A Detection-Guided Framework for Scalable Cell Representation Learning from Histopathology Images**
> *Submitted to NeurIPS 2026*

## Overview

Existing pathology foundation models learn representations at the patch or slide level using fixed-size inputs, which limits their ability to provide reliable cell-level representations. CellDETR addresses this gap by treating detected nuclei — rather than fixed patches — as the basic unit of representation learning.

The framework introduces two key designs on top of Deformable DETR:

1. **Location–Feature Decouple Module.** The decoder is split into a *detection decoder* (for nuclei localization) and a *representation decoder* (for cell morphology features). An initialization layer separately projects center coordinates `(x, y)` and box size `(w, h)` so positional and morphological information are decoupled. The multi-head self-attention layer before deformable cross-attention is removed in the representation decoder to discourage cross-cell interaction.

2. **Box-Constrained Attention Module.** A `tanh` activation followed by per-box `(w, h)` scaling restricts deformable sampling locations to the predicted bounding box region:

   ```
   (x̂, ŷ) = (x, y) + 0.5 · tanh((Δx, Δy)) ⊙ (w, h)
   ```

   This concentrates attention on the target nucleus and suppresses contamination from background tissue and neighboring cells.

CellDETR additionally supports DINO-style **cell-level contrastive pretraining**, where perturbed local boxes serve as anchors and the original instance boxes serve as global views. The InfoNCE objective is optimized over instance-level positive/negative pairs with an attention mask that prevents anchor-to-anchor leakage.

## Highlights

- **Supervised cell classification** on PanNuke that matches or surpasses Mask-RCNN, Micro-Net, HoverNet, CellViT, and Deformable DETR baselines.
- **Self-supervised pretraining** on 64,000 unlabeled H&E patches across three tissue types, with full fine-tuning yielding the best F1 scores on neoplastic, epithelial, inflammatory, and connective cells.
- **Spatial-transcriptomics-informed pretraining** with 21,000 Xenium-paired patches (~380K cells, 10 ST-derived cell types), enabling biologically grounded supervision and competitive cross-dataset transfer to PanNuke.
- **Two training modes**: *sequential* (predicted boxes feed the representation decoder end-to-end) and *parallel* (ground-truth boxes during training, predicted boxes at inference).

## Repository Structure

```
DecoupleDETR/
├── config/                # YAML configs for backbones, data, models, losses, experiments
│   ├── backbone/          # ResNet-50 / Swin-L, 4-level feature configs
│   ├── base/              # Loss and matcher configs
│   ├── data/              # PanNuke / CoNSeP / contrastive / semi / weak datasets
│   ├── experiments/       # Top-level experiment recipes
│   └── model/             # Decouple / contrastive / noised / semi / weak model configs
├── dataset/               # Data loading and augmentation
│   ├── pannuke.py         # PanNuke dataset
│   ├── consep.py          # CoNSeP dataset
│   ├── monuseg.py         # MoNuSeg dataset
│   ├── weak_dataset.py    # Unlabeled WSI patches for contrastive pretraining
│   ├── semi_dataset.py    # Xenium ST-derived semi-supervised dataset
│   └── transforms.py      # torchvision.transforms.v2 pipeline (incl. HEDJitter)
├── models/
│   ├── backbone/          # ResNet / Swin / position encoding
│   ├── deformable_detr/   # Deformable DETR baseline + custom CUDA ops
│   │   └── ops/           # MSDeformAttn and RestrictedMSDeformAttn ops
│   └── decouple_detr/     # CellDETR: decoupled decoder + box-constrained attention
├── trainer/
│   ├── supervised_trainer/   # Supervised training on PanNuke / CoNSeP
│   ├── contrastive_trainer/  # DINO-style cell-level contrastive pretraining
│   ├── semi_trainer/         # ST-informed semi-supervised training
│   └── weak_trainer/         # Weak-supervised training
├── experiments/           # Standalone experiment entry points (sequence / parallel / efficiency)
├── eval/                  # PanNuke evaluation protocol
└── util/                  # Box ops, distributed, config (YAML w/ __file__ includes), misc
```

## Installation

### Environment

- Python 3.9+
- PyTorch 1.13+ with CUDA 11.7+
- `torchvision >= 0.15` (required for `torchvision.transforms.v2`)

### Dependencies

```bash
pip install -r requirements.txt
```

### Build the Deformable Attention CUDA ops

```bash
cd models/deformable_detr/ops
sh make.sh
# verify
python test.py
```

The custom `RestrictedMSDeformAttn` operator implementing the box-constrained sampling lives in `models/deformable_detr/ops/modules/restricted_ms_deform_attn.py`.

## Datasets

| Dataset | Use | Notes |
|---|---|---|
| **PanNuke** | Supervised benchmark | 5 cell types: neoplastic, epithelial, inflammatory, connective, necrotic. Boundaries are converted to bounding boxes. |
| **Unlabeled WSI patches** | Self-supervised pretraining | 64,000 H&E patches (256×256) from three tissue types with nucleus segmentation but no cell-type labels. |
| **Xenium WSI–ST paired data** | ST-informed pretraining | 21,000 patches with ~380K cells annotated into 10 ST-derived cell types via Tacco. |

Place dataset roots and update the path fields in `config/data/*.yaml` accordingly. Folds for PanNuke are configured under `dataset.{train,val,test}.fold` (e.g. `fold1`, `fold2`, `fold3`) for cross-validation.

## Training

All entry points consume a YAML config tree assembled with the project's `ConfigDict` loader (see `util/config/`). Configs support `__file__` includes for composition.

### Supervised training on PanNuke

```bash
python -m trainer.supervised_trainer.main
# uses config/experiments/pannuke_supervised.yaml by default
```

The default config trains for 100 epochs with ResNet-50 (4-level features), 900 queries, the decoupled representation decoder, and the box-constrained attention enabled (`restrict_attn: True`). The `__main__` block of `trainer/supervised_trainer/main.py` runs the three PanNuke fold permutations.

### Self-supervised contrastive pretraining

```bash
python -m trainer.contrastive_trainer.main
# config/experiments/contrastive_train.yaml
```

The contrastive trainer perturbs each ground-truth box following the noise model in the supplementary (Bernoulli mask + uniform scaling on `(w, h)` and `(cx, cy)`), produces local/global views per nucleus, and optimizes InfoNCE with a representation-decoder attention mask that prevents anchor-to-anchor leakage.

### ST-informed (Xenium) semi-supervised training

```bash
python -m trainer.semi_trainer.main
# config/experiments/semi_train.yaml
```

### Standalone experiment recipes

Sequential, parallel-with-noise, super-finetune, and weak-parallel variants are also provided as standalone entry points:

```bash
python -m experiments.suquence.main
python -m experiments.parallel_with_noise.main
python -m experiments.super_finetune.main
python -m experiments.weak_parallel.main
```

`experiments/efficiency_test/flop.ipynb` reproduces the FLOP / latency comparison.

## Evaluation

Evaluation follows the matching criterion and metric definitions in the PanNuke protocol. The implementation is in `eval/pannuke_eval.py`. For each cell type and for overall detection, precision, recall, and F1 are reported.

```python
from eval.pannuke_eval import evaluate
# called from the trainer engines after each evaluation epoch
```

Results reported in the paper:

- **Tab. 1** — Supervised PanNuke; CellDETR-S achieves the best detection F1 (0.82) and matches/beats baselines on all five cell types, with notable gains on necrotic cells (F1 0.43 vs 0.40 for Deformable DETR).
- **Tab. 2** — Self-supervised CellDETR; full fine-tuning improves F1 across neoplastic, epithelial, inflammatory, connective, and overall detection over the supervised baseline.
- **Tab. 3** — ST-informed pretraining; after PanNuke fine-tuning, performance is comparable to supervised SOTA, while zero-shot transfer demonstrates partial recognition for epithelial and inflammatory cells.

## Citation

```bibtex
@inproceedings{celldetr2026,
  title     = {CellDETR: A Detection-Guided Framework for Scalable Cell Representation Learning from Histopathology Images},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgements

This implementation builds on the open-source [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) codebase. We also thank the authors of PanNuke, CellViT, Xenium / Tacco, and DINO for releasing data and methods that made this work possible.

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) for details.
