import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from omegaconf import OmegaConf

from opensr_model import SRLatentDiffusion


# ---------------------------------------------------------
# Patch configuration
# ---------------------------------------------------------

PATCH_ID = "patch_03"

REPO = Path(__file__).resolve().parents[2]

PATCH_DIR = (
    REPO
    / "data/raw/test_patches/patch_03_agriculture/patch"
)

CHECKPOINT = (
    REPO
    / "experiments/ldsr_s2/opensr-ldsrs2_v1_0_0.ckpt"
)

OUTPUT_DIR = (
    REPO
    / f"data/outputs/benchmark/ldsr_s2/{PATCH_ID}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# Load Sentinel-2 patch
# ---------------------------------------------------------

band_files = [
    PATCH_DIR / "B02.tif",
    PATCH_DIR / "B03.tif",
    PATCH_DIR / "B04.tif",
    PATCH_DIR / "B08.tif",
]

bands = []

for path in band_files:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        bands.append(data)

stack = np.stack(bands, axis=0)

print("Input shape:", stack.shape)
print("Input dtype:", stack.dtype)

# Sentinel-2 L2A scaling
stack = stack / 10000.0

lr = torch.from_numpy(stack).unsqueeze(0).to(device)


# ---------------------------------------------------------
# Load LDSR-S2
# ---------------------------------------------------------

config_path = (
    Path(__import__("opensr_model").__file__).parent
    / "configs"
    / "config_10m.yaml"
)

config = OmegaConf.load(config_path)

model = SRLatentDiffusion(
    config,
    device=device
)

print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

weights = checkpoint["state_dict"]

# Match official loader
for key in list(weights.keys()):
    if "loss" in key:
        del weights[key]

model.model.load_state_dict(
    weights,
    strict=True
)

model.eval()

print("LDSR-S2 weights loaded.")


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

print("Running LDSR-S2...")

with torch.no_grad():
    sr = model.forward(
        lr,
        sampling_steps=100
    )

sr = sr.squeeze(0).detach().cpu().numpy()

print("Output shape:", sr.shape)
print("Output dtype:", sr.dtype)


# ---------------------------------------------------------
# Save GeoTIFF
# ---------------------------------------------------------

with rasterio.open(PATCH_DIR / "B02.tif") as src:
    transform = src.transform
    crs = src.crs

# 4× spatial resolution: 10m -> 2.5m
new_transform = rasterio.Affine(
    transform.a / 4,
    transform.b,
    transform.c,
    transform.d,
    transform.e / 4,
    transform.f,
)

output_path = (
    OUTPUT_DIR / f"ldsr_s2_{PATCH_ID}.tif"
)

profile = {
    "driver": "GTiff",
    "height": sr.shape[1],
    "width": sr.shape[2],
    "count": 4,
    "dtype": "float32",
    "crs": crs,
    "transform": new_transform,
}

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(sr.astype(np.float32))


# ---------------------------------------------------------
# Final information
# ---------------------------------------------------------

print("\nOutput saved:")
print(output_path)

print("\nFinal result:")
print("Input :", stack.shape)
print("Output:", sr.shape)
print("Resolution: 10m -> 2.5m")
print("CRS:", crs)

print(f"\n🔥 {PATCH_ID.upper()} LDSR-S2 INFERENCE COMPLETE 🔥")