# SIH26142 — Deep Learning Based Super Resolution Mapping

Deep-learning-based Super Resolution Mapping (SRM) for Sentinel-2 satellite imagery, targeting enhancement from **10 m to 2.5 m spatial resolution** while preserving spatial and spectral consistency.

---

## 🚀 Problem Statement

**SIH26142 — Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries**

The objective is to develop a deep-learning-based super-resolution pipeline that enhances medium-resolution satellite imagery while maintaining geographic consistency, spectral fidelity, and useful spatial detail.

The project focuses primarily on **Sentinel-2 10 m imagery → 2.5 m super-resolved imagery**.

---

## 🛰️ System Pipeline

```text
Sentinel-2 L2A Imagery (10 m)
            │
            ▼
      Preprocessing
            │
            ▼
   Super-Resolution Model
            │
            ▼
   Enhanced Imagery (2.5 m)
            │
            ▼
 Quality / Uncertainty Analysis
            │
            ▼
 Downstream Geospatial Analysis
            │
            ▼
 Natural-Language Query / Visualization
```

The core component of the system is the **super-resolution pipeline**. Additional AI-assisted geospatial analysis is built on top of the generated high-resolution imagery.

---

## 🎯 Project Objectives

- Enhance Sentinel-2 imagery from **10 m to 2.5 m** spatial resolution.
- Compare multiple pretrained satellite super-resolution models.
- Evaluate model behavior across different geographical and land-cover environments.
- Preserve spectral and geographic consistency during enhancement.
- Quantitatively evaluate reconstruction quality using paired LR-HR datasets.
- Analyze uncertainty and failure cases.
- Investigate whether super-resolution improves downstream geospatial tasks.
- Provide an interface for interacting with and analyzing the resulting imagery.

---

## 🧠 Models

The project evaluates multiple pretrained super-resolution models using a common benchmarking protocol.

### Currently Evaluated

- **LDSR-S2**
- **SEN2SR**

Additional pretrained super-resolution models will be evaluated and compared before selecting the final production model.

### Model Selection Strategy

Models are first screened on the same real-world Sentinel-2 patches for:

- Visual quality
- Spatial detail enhancement
- Spectral/color stability
- Artifact generation
- Robustness to difficult imagery
- Inference time
- GPU memory usage

Final model selection is performed using quantitative evaluation on paired LR-HR datasets.

---

## 📊 Benchmark Strategy

The evaluation is divided into two complementary layers.

### Layer 1 — Real-World Robustness

Five real Sentinel-2 patches are used to examine model behavior on different scene types:

| Patch | Environment |
|---|---|
| Patch 01 | Forest |
| Patch 02 | Peri-urban / Mixed |
| Patch 03 | Agriculture |
| Patch 04 | Atmospheric / Cloud Stress |
| Patch 05 | Dense Urban / Infrastructure |

These patches represent actual Sentinel-2 observations and do **not** necessarily have corresponding high-resolution ground truth.

Therefore, they are used for:

- Qualitative comparison
- Artifact inspection
- Robustness screening
- Runtime comparison
- GPU memory comparison
- Failure-case analysis

Metrics such as PSNR and SSIM are **not claimed on these real-world patches without appropriate high-resolution ground truth**.

### Layer 2 — Paired LR-HR Evaluation

Paired datasets containing corresponding low-resolution and high-resolution imagery are used for quantitative evaluation.

Planned metrics include:

- PSNR
- SSIM
- SAM
- Spatial-detail metrics
- Spectral consistency
- Inference time
- GPU memory usage

The final selected model is determined using both reconstruction quality and real-world robustness.

---

## 🗺️ Data

Raw satellite imagery and generated outputs are intentionally excluded from Git because of their size.

Expected local structure:

```text
data/
├── raw/
│   └── test_patches/
└── outputs/
```

### Raw Data

Place Sentinel-2 imagery and benchmark patches under:

```text
data/raw/
```

### Generated Outputs

Model predictions, comparison images, and other generated artifacts are stored under:

```text
data/outputs/
```

Both directories are ignored by Git.

---

## ⚙️ Installation

### Requirements

- Python **3.12**
- NVIDIA GPU recommended for practical inference
- CUDA-compatible PyTorch installation

Create the project environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### Tested Development Environment

The current development environment has been tested with:

| Component | Version |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.14.0 + CUDA 13.0 |
| CUDA | 13.0 |
| GPU | NVIDIA GeForce RTX 2080 Super |
| opensr-model | 1.1.1 |
| opensr-utils | 2.0.0 |
| sen2sr | 0.8.5 |
| rasterio | 1.5.1 |
| NumPy | 2.5.2 |

> **Note:** PyTorch should be installed according to the CUDA and GPU environment of the target machine. The versions above describe the currently tested development setup.

---

## 📦 Model Weights

Large model checkpoints are **not stored in this repository**.

Each model's weights should be downloaded separately according to its installation instructions.

### LDSR-S2

The LDSR-S2 checkpoint is expected locally at:

```text
experiments/ldsr_s2/opensr-ldsrs2_v1_0_0.ckpt
```

The checkpoint is ignored by Git because of its size.

### SEN2SR

The currently tested SEN2SR model files are stored under:

```text
experiments/models/sen2sr/
```

The current configuration corresponds to the **SEN2SR main CNN / SEN2SRLite RGB-NIR 4× model**.

SEN2SR expects the RGB-NIR bands in the following order:

```text
B04, B03, B02, B08
```

---

## 🛰️ Sentinel-2 Input

The primary input bands used by the current RGB-NIR models are:

| Band | Description | Resolution |
|---|---|---|
| B02 | Blue | 10 m |
| B03 | Green | 10 m |
| B04 | Red | 10 m |
| B08 | Near Infrared | 10 m |

Sentinel-2 L2A surface reflectance data is used for the current experiments.

Input data is normalized from Sentinel-2 digital-number scaling using the appropriate model preprocessing.

> **Important:** Different models may require different band ordering or preprocessing. These requirements must be followed exactly for each model.

---

## 🧪 Current Benchmark Workflow

The current benchmark workflow is:

```text
1. Select common Sentinel-2 test patches
                ↓
2. Prepare model-specific input
                ↓
3. Run each pretrained model
                ↓
4. Generate 2.5 m outputs
                ↓
5. Compare spatial and spectral behavior
                ↓
6. Record runtime / GPU memory
                ↓
7. Evaluate finalists on paired LR-HR datasets
                ↓
8. Select final production model
```

The same physical test patches are used across models wherever their input/output specifications allow.

---

## 🔬 Uncertainty and Failure Analysis

A major focus of the project is understanding **when super-resolution can be trusted**.

The system will investigate:

- Model uncertainty
- Spectral inconsistencies
- Hallucinated spatial details
- Atmospheric and cloud contamination
- Model-specific artifacts
- Failure cases across different environments

For example, difficult atmospheric/cloud-contaminated imagery is treated as a robustness stress case rather than as clean evidence of spatial reconstruction quality.

---

## 🌍 Downstream Geospatial Utility

Super-resolution quality is not evaluated only by visual sharpness.

The project also investigates:

> **Does super-resolution improve downstream geospatial decision-making?**

Potential downstream applications include:

- Building and road analysis
- Agricultural analysis
- Land-cover analysis
- Water-body analysis
- Change detection
- Vegetation analysis

The purpose is to determine whether enhanced imagery provides useful information for downstream tasks rather than simply producing visually sharper images.

---

## 🤖 AI-Assisted Geospatial Interaction

The final system is intended to provide an AI-assisted interface on top of the super-resolved imagery.

Planned capabilities include:

- Natural-language queries
- Visual question answering
- Change analysis
- Geospatial reasoning
- Evidence and confidence reporting
- Visualization of relevant regions

These components are designed to operate **on top of the core super-resolution pipeline** rather than replacing it.

---

## 📁 Repository Structure

```text
sih-satellite/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── SIH26142.ipynb
│
├── .vscode/
│   └── settings.json
│
├── data/
│   ├── raw/                  # Ignored — satellite data
│   └── outputs/              # Ignored — generated results
│
└── experiments/
    │
    ├── benchmarks/
    │   ├── run_ldsr_patch.py
    │   ├── run_sen2sr_all.py
    │   ├── run_sen2sr_patch.py
    │   ├── compare_all_sen2sr.py
    │   └── ...
    │
    ├── ldsr_s2/
    │   ├── infer_reals2.py
    │   ├── test_ldsr.py
    │   ├── compare_real.py
    │   └── *.ckpt             # Ignored — model weights
    │
    └── models/
        └── sen2sr/
            ├── model.py
            ├── predictor.py
            ├── config.json
            └── ...
```

---

## 🚧 Project Status

### Completed

- [x] Sentinel-2 L2A data acquisition
- [x] Real-world benchmark patch selection
- [x] LDSR-S2 inference pipeline
- [x] SEN2SR inference pipeline
- [x] Initial LDSR-S2 vs SEN2SR qualitative comparison
- [x] Atmospheric/cloud stress-case analysis
- [x] Reproducible project environment setup

### In Progress

- [ ] Evaluate additional pretrained SR models
- [ ] Quantitative paired LR-HR evaluation
- [ ] Final model selection
- [ ] Uncertainty estimation
- [ ] Downstream geospatial evaluation
- [ ] AI-assisted geospatial interface
- [ ] End-to-end production pipeline

---

## 📌 Important Reproducibility Notes

- Raw satellite imagery is not committed to Git.
- Generated outputs are not committed to Git.
- Large model checkpoints are not committed to Git.
- Model-specific preprocessing and band ordering must be respected.
- Quantitative metrics must only be reported when appropriate high-resolution ground truth is available.
- Real-world Sentinel-2 patches are used primarily for robustness and qualitative screening.
- Final model selection should be based on both quantitative reconstruction quality and real-world robustness.

---

## 📄 License

See [`LICENSE`](LICENSE) for the project license.