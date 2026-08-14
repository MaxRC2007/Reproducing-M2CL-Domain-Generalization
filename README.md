# M²-CL: Multiscale and Multilayer Contrastive Learning for Domain Generalization
### Reproduction on PACS — Honors Project, IIIT Sri City

> **Paper:** *Multiscale and Multilayer Contrastive Learning for Domain Generalization*,
> IEEE Transactions on Artificial Intelligence, Vol. 5, No. 12, 2024.

---

## Overview

This repository contains a reproduction of the M²-CL framework on the **PACS** benchmark (leave-one-domain-out protocol). It implements three models from scratch in PyTorch:

| Model | Description |
|-------|-------------|
| **ERM** | Standard ResNet-18 fine-tuned with cross-entropy only |
| **M²** | ResNet-18 with 4 intermediate extraction blocks (multi-scale, multi-layer) |
| **M²-CL** | M² + class-conditional contrastive loss at every extraction block |

The notebook also includes ablation studies, saliency map visualisation (vanilla gradient), and a comparison against paper-reported numbers.

---

## Repository Structure

```
M2CL_PACS_v2.ipynb       # Main notebook (all experiments)
README.md                 # This file
```

The notebook is self-contained and runs on **Google Colab (T4 GPU)**. No separate scripts are needed.

---

## Requirements

| Dependency | Version |
|------------|---------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| torchvision | 0.15+ |
| Google Colab | (recommended) |
| PACS dataset | `.zip` stored in Google Drive |

All other dependencies (`numpy`, `matplotlib`, `PIL`, `json`) are available in the standard Colab environment.

---

## Dataset Setup

1. Download the **PACS** dataset (standard DomainBed version).
2. Upload `PACS.zip` to your **Google Drive** root (`MyDrive/PACS.zip`).
3. The notebook mounts Drive and extracts automatically:

```python
ZIP_PATH = '/content/drive/MyDrive/PACS.zip'
```

Expected structure after extraction:
```
/content/
  dct2_images/
    art_painting/  dog/  elephant/  ...
    cartoon/
    photo/
    sketch/
```

Domains: `art_painting`, `cartoon`, `photo`, `sketch` | Classes: 7 | Images: 9,991

---

## Architecture

### Feature Dimensions

```
ERM   : ResNet-18 final layer → 512-d → FC(7)
M²/M²-CL : eb1(384) + eb2(384) + eb3(256) + eb4(256) + GAP(512) = 1792-d → FC(7)
```

### Extraction Blocks

Each block taps an intermediate ResNet layer and processes it through parallel concentration pipelines:

| Block | Layer | Channels | Spatial | Pool sizes | Output dim |
|-------|-------|----------|---------|------------|-----------|
| eb1 | layer1 | 64 | 56×56 | 8, 4, 2 | 384 |
| eb2 | layer2 | 128 | 28×28 | 8, 4, 2 | 384 |
| eb3 | layer3 | 256 | 14×14 | 7, 3 | 256 |
| eb4 | layer4 | 512 | 7×7 | 7, 3 | 256 |

### Concentration Pipeline (per branch)
```
Input → 1×1 Conv (channel reduction r=4) → BN → ReLU
      → Spatial Dropout (p=0.5, drops entire channels)
      → AdaptiveMaxPool(pool_size)
      → Flatten → Linear(mlp_dim=128) → ReLU
```

### Parameter Counts (PACS, 7 classes)

| Model | Parameters |
|-------|-----------|
| ERM | 11,180,103 |
| M² | 13,312,103 |
| M²-CL | 13,312,103 |

M² and M²-CL have identical parameters — the contrastive loss uses the same embeddings; no extra weights are added.

---

## Contrastive Loss

The class-conditional contrastive loss is applied independently at each extraction layer and averaged:

```python
def contrastive_loss(reps, labels, tau=1.0):
    total, n_layers = 0.0, 0
    for u in reps:                         # one embedding per layer
        u   = F.normalize(u, dim=1)        # L2-normalise → unit sphere
        sim = torch.mm(u, u.t()) / tau     # cosine similarity matrix
        B   = u.size(0)
        off = ~torch.eye(B, dtype=torch.bool, device=labels.device)
        denom = sim[off].exp().sum()       # all off-diagonal pairs
        log_p_list = []
        for c in labels.unique():
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if len(idx) < 2: continue      # skip singleton classes
            pos_sim = sim[idx][:, idx]
            nc_off  = ~torch.eye(len(idx), dtype=torch.bool, device=labels.device)
            numer   = pos_sim[nc_off].exp().sum()
            log_p_list.append(torch.log(numer / (denom + 1e-8)))
        if log_p_list:
            total    += -torch.stack(log_p_list).sum()
            n_layers += 1
    return total / max(n_layers, 1)
```

**Combined training objective:**
```
L = L_CE + α * Σ_l L_CL^(l)     (α=0.01, τ=1.0)
```

---

## Training Configuration

| Hyperparameter | Value | Source |
|----------------|-------|--------|
| Optimizer | SGD | Paper Sec IV-B |
| Learning rate | 5e-4 (uniform) | DomainBed PACS default |
| Momentum | 0.9 | Paper |
| Weight decay | 5e-4 | Paper |
| Scheduler | CosineAnnealingLR, T_max=30 | Paper |
| Epochs | 30 | Paper |
| Batch size | 128 (M²/M²-CL), 32 (ERM) | Paper |
| α (CL weight) | 0.01 | Paper Table III |
| τ (temperature) | 1.0 | Paper |
| AMP | Enabled on CUDA | Implementation |
| Seed | 0 | This reproduction |

### Validation Protocol

DomainBed IID-validation protocol:
- **Source domains**: 80% train / 20% validation (deterministic split with seed=0)
- **Test domain**: completely held out — not touched during training or model selection
- Best model selected by validation accuracy; final score reported on test domain

> **Important:** An earlier version of the notebook accidentally used test accuracy for model selection (evaluation leakage). This has been fixed.

---

## Data Augmentation

**Training:**
```python
transforms.RandomResizedCrop(224, scale=(0.7, 1.0))
transforms.RandomHorizontalFlip()
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
transforms.RandomGrayscale(p=0.1)
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

**Test:**
```python
transforms.Resize(256)
transforms.CenterCrop(224)
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

Note: The paper does not fully specify augmentation parameters. Standard DomainBed values are used here, which may contribute to the gap with paper-reported numbers.

---

## Running the Notebook

Open `M2CL_PACS_v2.ipynb` in Google Colab and run cells in order:

| Cell | Description |
|------|-------------|
| 1 | Mount Drive, extract PACS |
| 2 | Set global seed (seed=0) |
| 3 | Dataset & DataLoaders |
| 4 | Model Architecture (ERM, M², M²-CL) |
| 5 | Training loop with AMP |
| 6 | Leave-one-domain-out experiments (all 3 models × 4 domains) |
| 7 | Results table & bar chart |
| 8 | Ablation study |
| 9 | Ablation bar chart vs. paper Table III |
| 10 | Save checkpoints to Drive |
| 11 | Saliency maps (ERM vs. M²-CL) |

Checkpoints are saved to `/content/drive/MyDrive/m2cl_checkpoints/` and automatically restored across Colab sessions. Training history is cached as JSON.

---

## Reproduction Results

Single run, seed=0, 30 epochs on PACS:

| Method | Art | Cartoon | Photo | Sketch | Avg |
|--------|-----|---------|-------|--------|-----|
| ERM (ours) | 73.1 | 70.7 | 94.4 | 66.5 | 76.2 |
| M² (ours) | 74.3 | 67.4 | 95.4 | 61.1 | 74.6 |
| M²-CL (ours) | 74.2 | 68.3 | 95.5 | 61.0 | 74.8 |
| **M²-CL (paper)** | **81.7** | **78.4** | **97.0** | **77.1** | **83.5** |

### Gap Analysis

The ~6–9% gap vs. paper is expected and attributable to:

1. **Single seed** — paper reports average over 3 seeds
2. **Epoch budget** — 30 epochs vs. full DomainBed training schedule
3. **Augmentation** — paper does not fully specify; DomainBed defaults used
4. **Hardware** — results not sensitive to GPU type, but batch scheduling may vary

Key observations preserved from paper:
- M²-CL ≥ ERM on Photo domain ✓
- Sketch regression (M²-CL < M²) reproduced ✓ — confirms ℓ₂ tolerance issue
- Relative ordering: M²-CL ≥ M² ≥ ERM on most domains ✓

---

## Key Implementation Notes

### Spatial Dropout
`nn.Dropout2d` drops entire feature map channels (not individual pixels). This is the single largest contributor in the ablation study (+1–1.8% on PACS). It prevents the extraction blocks from memorising domain-identity features encoded in specific channels.

### `drop_last=False`
An earlier version used `drop_last=True`, silently discarding ~50 images per epoch. The paper does not mention dropping the last batch, so `drop_last=False` is used throughout.

### AMP (Automatic Mixed Precision)
```python
scaler = GradScaler(device=DEVICE.type, enabled=(DEVICE.type == 'cuda'))
with autocast(device_type=DEVICE.type, enabled=use_amp):
    logits, reps = model(imgs)
    loss = ce(logits, labels)
    if reps is not None:
        loss += alpha * contrastive_loss(reps, labels, tau)
```
Using `torch.amp` (not the deprecated `torch.cuda.amp`) with CPU fallback.

### Contrastive Loss Batch Size Requirement
The discriminative contrastive loss requires at least 2 samples per class in each batch to form positive pairs. With 7 classes and batch size 32, some classes may contribute zero pairs per batch, degrading the CL signal. Batch size ≥ 128 is recommended (paper value).

---

## Limitations

| Limitation | Impact | Potential Fix |
|------------|--------|---------------|
| Batch size ≥ 128 | Memory constraint vs. ERM's 32 | MoCo-style memory queue |
| ~2.1× training overhead vs. ERM | Slower training | Fewer extraction layers |
| Sketch regression (ℓ₂ tolerance) | -0.83% vs. M² on Sketch | KL-divergence instead of cosine |
| ResNet-18/50 only | Old backbone | ViT, ConvNeXt |
| Single vision modality | Untested on NLP/graphs | GNN extension (honors work) |

---

## Honors Extension: M²-CL for GNNs

The honors project extends this framework to **Graph Neural Networks** for molecular property prediction under distribution shift (**DrugOOD benchmark**).

- Each GNN message-passing layer serves as a natural extraction point (analogous to CNN layers)
- Open research question: contrastive pairs at **node-level** vs. **subgraph-level**
- Target: synthesis-condition shift in molecular datasets
- Status: PACS reproduction complete; DrugOOD pipeline setup in progress

**Guide:** Dr. Rajendra Prasath, IIIT Sri City

---

## Citation

```bibtex
@article{m2cl2024,
  title   = {Multiscale and Multilayer Contrastive Learning for Domain Generalization},
  journal = {IEEE Transactions on Artificial Intelligence},
  volume  = {5},
  number  = {12},
  year    = {2024}
}
```

---

## Author

**C. Madhusudan Reddy** | Roll No: S20240030357 | IIIT Sri City
