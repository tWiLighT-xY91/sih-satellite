import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

# Add cloned Satlas repo to Python path
SATLAS_REPO = Path.home() / "sih" / "satlas-super-resolution"
sys.path.insert(0, str(SATLAS_REPO))

from ssr.archs.rrdbnet_arch import SSR_RRDBNet


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PATCH_ID = "patch_01"

PROJECT_ROOT = Path.home() / "sih" / "sih-satellite"
PATCH_DIR = PROJECT_ROOT / "data" / "raw" / "test_patches" / "patch_01_forest" / "patch"

OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "benchmark" / "satlas" / PATCH_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = SATLAS_REPO / "esrgan_1S2.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Satlas 1-image model = 3 RGB channels
# Their released 1S2 model is x4.
MODEL = SSR_RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4,
)


# ---------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

print("Checkpoint type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("Checkpoint keys:", list(checkpoint.keys())[:20])


# Try common Satlas/BasicSR checkpoint structures
if isinstance(checkpoint, dict) and "params_ema" in checkpoint:
    state_dict = checkpoint["params_ema"]
elif isinstance(checkpoint, dict) and "params" in checkpoint:
    state_dict = checkpoint["params"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint


# Remove common prefixes if present
clean_state_dict = {}

for key, value in state_dict.items():
    if key.startswith("module."):
        key = key[len("module."):]

    if key.startswith("generator."):
        key = key[len("generator."):]

    clean_state_dict[key] = value


missing, unexpected = MODEL.load_state_dict(
    clean_state_dict,
    strict=False,
)

print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

MODEL.to(DEVICE)
MODEL.eval()


# ---------------------------------------------------------
# Find input files
# ---------------------------------------------------------

def find_band(name):
    candidates = [
        PATCH_DIR / f"{name}.jp2",
        PATCH_DIR / f"{name}_10m.jp2",
        PATCH_DIR / f"TILE_{name}.jp2",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find {name} in {PATCH_DIR}"
    )


# RGB order for Satlas:
# Red = B04
# Green = B03
# Blue = B02

band_paths = {
    "B04": find_band("B04"),
    "B03": find_band("B03"),
    "B02": find_band("B02"),
}

print("Input bands:")
for band, path in band_paths.items():
    print(f"  {band}: {path}")


# ---------------------------------------------------------
# Read Sentinel-2
# ---------------------------------------------------------

arrays = []

profile = None

for band in ["B04", "B03", "B02"]:
    path = band_paths[band]

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)

        if profile is None:
            profile = src.profile.copy()

    arrays.append(arr)


rgb = np.stack(arrays, axis=0)

print("Raw RGB shape:", rgb.shape)
print("Raw range:", float(rgb.min()), float(rgb.max()))


# ---------------------------------------------------------
# Sentinel-2 L2A DN -> reflectance
# ---------------------------------------------------------

rgb = rgb / 10000.0

rgb = np.clip(rgb, 0.0, 1.0)

print("Reflectance range:")
print(
    "  min =",
    float(rgb.min()),
    "max =",
    float(rgb.max()),
)


# ---------------------------------------------------------
# Satlas expects RGB input.
#
# For the released 1S2 model, the repository's published
# preprocessing uses 8-bit RGB-style input normalization.
#
# We therefore convert reflectance to 0-255 and normalize.
# ---------------------------------------------------------

rgb_8bit = np.clip(rgb * 255.0, 0, 255)

tensor = torch.from_numpy(rgb_8bit / 255.0)

tensor = tensor.unsqueeze(0).float().to(DEVICE)

print("Model input:", tensor.shape)


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

with torch.no_grad():
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(DEVICE.type == "cuda"),
    ):
        output = MODEL(tensor)


output = output.squeeze(0).float().cpu().numpy()

print("Output shape:", output.shape)
print(
    "Output range before clipping:",
    float(output.min()),
    float(output.max()),
)


# ---------------------------------------------------------
# Clip output
# ---------------------------------------------------------

output = np.clip(output, 0.0, 1.0)


# ---------------------------------------------------------
# Write GeoTIFF
# ---------------------------------------------------------

height, width = output.shape[1:]

profile.update(
    height=height,
    width=width,
    count=3,
    dtype="float32",
)

# 10 m -> 2.5 m
transform = profile["transform"]

profile["transform"] = rasterio.Affine(
    transform.a / 4,
    transform.b,
    transform.c,
    transform.d,
    transform.e / 4,
    transform.f,
)

output_path = OUTPUT_DIR / f"satlas_{PATCH_ID}.tif"

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(output.astype(np.float32))


print()
print("DONE")
print("Output:", output_path)
print("Final shape:", output.shape)
print("Final range:", float(output.min()), float(output.max()))