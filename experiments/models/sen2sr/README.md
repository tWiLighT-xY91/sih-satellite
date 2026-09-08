---
license: cc0-1.0
base_model: tacofoundation/SEN2SR
tags:
- sentinel-2
- super-resolution
- remote-sensing
- pytorch
pipeline_tag: image-to-image
model-index:
  - name: WEO-SAS/sen2sr
    results:
      - task:
          type: image-to-image
          name: Satellite Image Super-Resolution
        dataset:
          name: NAIP (4x Sentinel-2 → NAIP)
          type: custom
          config: naip
        metrics:
          - name: Improvement Score
            type: improvement
            value: 0.8743
          - name: Hallucination Rate
            type: hallucination
            value: 0.0561
          - name: Omission Rate
            type: omission
            value: 0.0696
      - task:
          type: image-to-image
          name: Satellite Image Super-Resolution
        dataset:
          name: SPOT (4x Sentinel-2 → SPOT)
          type: custom
          config: spot
        metrics:
          - name: Improvement Score
            type: improvement
            value: 0.7992
          - name: Hallucination Rate
            type: hallucination
            value: 0.0734
          - name: Omission Rate
            type: omission
            value: 0.1274
      - task:
          type: image-to-image
          name: Satellite Image Super-Resolution
        dataset:
          name: Spain Crops (4x Sentinel-2 → SPOT)
          type: custom
          config: spain_crops
        metrics:
          - name: Improvement Score
            type: improvement
            value: 0.8406
          - name: Hallucination Rate
            type: hallucination
            value: 0.0735
          - name: Omission Rate
            type: omission
            value: 0.0859
      - task:
          type: image-to-image
          name: Satellite Image Super-Resolution
        dataset:
          name: Spain Urban (4x Sentinel-2 → SPOT)
          type: custom
          config: spain_urban
        metrics:
          - name: Improvement Score
            type: improvement
            value: 0.6954
          - name: Hallucination Rate
            type: hallucination
            value: 0.1156
          - name: Omission Rate
            type: omission
            value: 0.1890
---

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/G1o2oiRwJaqw4ZP9nG0NO.webp" width="100%">
</p>

<p align="center">
  <em>Sentinel-2 super-resolution up to 2.5 m — WEO-SAS packaging of <a href="https://huggingface.co/tacofoundation/SEN2SR">tacofoundation/SEN2SR</a></em>
</p>

---

This repository re-packages the original [tacofoundation/SEN2SR](https://huggingface.co/tacofoundation/SEN2SR) models with the **WEO-SAS standard interface** (`model.py`, `predictor.py`, `config.json`) so they can be loaded and used identically to all other WEO-SAS models.

**Original work:** [ESAOpenSR/sen2sr](https://github.com/ESAOpenSR/sen2sr) — license CC0-1.0.

---

## Model Variants

Six variants are available as **HuggingFace branches**, each with a different architecture, input bands, and upscaling factor.

| Branch | Architecture | Input bands | Output bands | Scale | Description |
|---|---|---|---|---|---|
| `main` *(default)* | CNN | 4 (RGBN) | 4 (RGBN) | 4× | SEN2SRLite — RGBN 10 m → 2.5 m |
| `lite-rswir-x2` | CNN | 10 (all S2) | 6 (RSWIR) | 2× | SEN2SRLite — 20 m bands → 10 m |
| `lite-main` | CNN | 10 (all S2) | 10 (all S2) | 4× | SEN2SRLite — full 10-band pipeline 10 m → 2.5 m |
| `mamba-rgbn-x4` | Mamba | 4 (RGBN) | 4 (RGBN) | 4× | SEN2SR — RGBN 10 m → 2.5 m (higher accuracy) |
| `mamba-rswir-x2` | Swin2SR | 10 (all S2) | 6 (RSWIR) | 2× | SEN2SR — 20 m bands → 10 m (higher accuracy) |
| `mamba-main` | Mamba + Swin2SR | 10 (all S2) | 10 (all S2) | 4× | SEN2SR — full 10-band pipeline (highest accuracy) |

**Band order expected as input:**

| Variant | Bands |
|---|---|
| RGBN (`main`, `mamba-rgbn-x4`) | B04, B03, B02, B08 |
| All others (10 bands) | B04, B03, B02, B08, B05, B06, B07, B8A, B11, B12 |

---

## Installation

```bash
# For CNN variants (main, lite-rswir-x2, lite-main)
pip install sen2sr safetensors huggingface_hub rasterio

# For Mamba/Swin variants (mamba-*)
pip install mamba-ssm --no-build-isolation
pip install sen2sr safetensors huggingface_hub rasterio
```

---

## Usage

All variants share the **same interface**. Only the `revision` argument changes.

### Load any variant

```python
from huggingface_hub import snapshot_download
import sys

# Choose your variant:
local_dir = snapshot_download("WEO-SAS/sen2sr")                            # RGBN 4x (CNN) — default
local_dir = snapshot_download("WEO-SAS/sen2sr", revision="lite-rswir-x2") # RSWIR 2x (CNN)
local_dir = snapshot_download("WEO-SAS/sen2sr", revision="lite-main")      # Full 10-band 4x (CNN)
local_dir = snapshot_download("WEO-SAS/sen2sr", revision="mamba-rgbn-x4") # RGBN 4x (Mamba)
local_dir = snapshot_download("WEO-SAS/sen2sr", revision="mamba-rswir-x2")# RSWIR 2x (Swin2SR)
local_dir = snapshot_download("WEO-SAS/sen2sr", revision="mamba-main")     # Full 10-band 4x (Mamba+Swin)

sys.path.insert(0, local_dir)
from model import Model

model = Model(local_dir=local_dir)
print(model.description)
```

### Array inference

```python
import numpy as np

# image: (C, H, W) float32, values in [0, 1]  (C=4 for RGBN, C=10 for full-band)
image = np.random.rand(4, 128, 128).astype("float32")

sr = model.predict(image)   # (C, H*4, W*4) float32
print(sr.shape)             # (4, 512, 512)
```

### GeoTIFF pipeline

Reads Sentinel-2 DN values directly (auto-normalises by /10000), writes a super-resolved GeoTIFF with the correct pixel size.

```python
model.predict_tif(
    input_path  = "s2_scene_10m.tif",
    output_path = "s2_scene_2p5m.tif",
    bands       = [0, 1, 2, 3],   # 0-based band indices (default: first C bands)
)
```

### Override config at load time

```python
model = Model(local_dir=local_dir, patch_size=256, overlap=64)
```

---

## RGBN 10 m → 2.5 m (`main`, `mamba-rgbn-x4`)

Super-resolves the four 10 m Sentinel-2 bands (Red, Green, Blue, NIR) by 4×.

<p align="center">
  <img src="https://huggingface.co/tacofoundation/SEN2SR/resolve/main/assets/srimg02.png" width="100%">
</p>

---

## Full 10-band 10 m → 2.5 m (`lite-main`, `mamba-main`)

Multi-stage pipeline: RGBN bands are super-resolved at 4×, while the 20 m bands (B05, B06, B07, B8A, B11, B12) are first sharpened to 10 m then to 2.5 m. All 10 bands are returned at 2.5 m.

<p align="center">
  <img src="https://huggingface.co/tacofoundation/SEN2SR/resolve/main/assets/srimg01.png" width="100%">
</p>

---

## RSWIR 20 m → 10 m (`lite-rswir-x2`, `mamba-rswir-x2`)

Sharpens the six 20 m Sentinel-2 bands (B05, B06, B07, B8A, B11, B12) to 10 m resolution using all 10 bands as context input.

<p align="center">
  <img src="https://huggingface.co/tacofoundation/SEN2SR/resolve/main/assets/srimg03.png" width="100%">
</p>

---

## Large image inference

For images larger than the 128×128 training patch size, `predict_tif` and `predict` automatically tile the input with overlapping patches and blend them seamlessly.

<p align="center">
  <img src="https://huggingface.co/tacofoundation/SEN2SR/resolve/main/assets/srimg05.png" width="95%">
</p>

---

## Repository structure

Each branch contains a flat directory with the same set of files:

```
config.json               # Variant-specific inference parameters
model.py                  # Public entry point (WEO-SAS standard)
predictor.py              # Tiled inference logic
sen2sr_pt.py              # HF-aware model loader (handles CNN / Mamba / Swin)
base.py                   # Abstract base class
model.safetensor          # Primary model weights
hard_constraint.safetensor# Hard-constraint weights
load.py                   # Original tacofoundation loading script
mlm.json                  # Original MLSTAC metadata
# multi-stage branches also include:
sr_model.safetensor / sr_hard_constraint.safetensor   (RGBN stage)
f2_model.safetensor / f2_hard_constraint.safetensor   (RSWIR 2x stage)
```

---

## Citation

If you use these models please cite the original work:

```bibtex
@software{sen2sr2024,
  author  = {Aybar, Cesar and others},
  title   = {SEN2SR: Sentinel-2 Super-Resolution},
  url     = {https://github.com/ESAOpenSR/sen2sr},
  year    = {2024}
}
```
