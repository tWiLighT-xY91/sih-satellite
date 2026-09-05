import torch
import rasterio
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf

import opensr_model


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "sentinel-2"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "ldsr_s2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


BAND_FILES = {
    "B02": next(DATA_DIR.glob("*_B02_10m.jp2")),
    "B03": next(DATA_DIR.glob("*_B03_10m.jp2")),
    "B04": next(DATA_DIR.glob("*_B04_10m.jp2")),
    "B08": next(DATA_DIR.glob("*_B08_10m.jp2")),
}


# ============================================================
# 2. Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# 3. Read Sentinel-2 bands
# ============================================================

print("\nLoading Sentinel-2 bands...")

bands = []
profile = None

for band_name in ["B02", "B03", "B04", "B08"]:

    path = BAND_FILES[band_name]

    print(f"  {band_name}: {path.name}")

    with rasterio.open(path) as src:

        data = src.read(1)

        if profile is None:
            profile = src.profile.copy()

        bands.append(data.astype(np.float32))

# Stack in model-required order:
# B02, B03, B04, B08

image = np.stack(bands, axis=0)

print(f"\nFull image shape: {image.shape}")
print(f"Data type: {image.dtype}")


# ============================================================
# 4. Find a valid 128 x 128 patch
# ============================================================

PATCH_SIZE = 128

print("\nSearching for valid 128 x 128 patch...")

height, width = image.shape[1:]

patch = None
row_start = None
col_start = None

# Search on a coarse grid rather than checking every pixel.
step = 256

for row in range(0, height - PATCH_SIZE + 1, step):

    for col in range(0, width - PATCH_SIZE + 1, step):

        candidate = image[
            :,
            row:row + PATCH_SIZE,
            col:col + PATCH_SIZE
        ]

        # Fraction of pixels where ALL four bands are > 0
        valid_mask = np.all(candidate > 0, axis=0)

        valid_fraction = valid_mask.mean()

        # Require at least 99% valid pixels
        if valid_fraction >= 0.99:

            patch = candidate
            row_start = row
            col_start = col

            print(
                f"Found patch at row={row_start}, "
                f"col={col_start}, "
                f"valid={valid_fraction:.2%}"
            )

            break

    if patch is not None:
        break


if patch is None:
    raise RuntimeError("Could not find a sufficiently valid 128x128 patch.")


# ============================================================
# 5. Convert Sentinel-2 scaled values to reflectance
# ============================================================

# Sentinel-2 L2A values are stored using a scale of 10,000.
patch = patch / 10000.0

print("\nPatch statistics after /10000 normalization:")

for i, band_name in enumerate(["B02", "B03", "B04", "B08"]):

    band = patch[i]

    print(
        f"  {band_name}: "
        f"min={band.min():.4f}, "
        f"median={np.median(band):.4f}, "
        f"max={band.max():.4f}"
    )


# ============================================================
# 6. Convert to PyTorch tensor
# ============================================================

lr = torch.from_numpy(patch).unsqueeze(0).to(device)

print(f"\nModel input shape: {lr.shape}")


# ============================================================
# 7. Load LDSR-S2 configuration
# ============================================================

package_dir = Path(opensr_model.__file__).parent

config_path = (
    package_dir
    / "configs"
    / "config_10m.yaml"
)

print(f"\nConfig: {config_path}")

config = OmegaConf.load(config_path)


# ============================================================
# 8. Create LDSR-S2
# ============================================================

print("\nCreating LDSR-S2 model...")

model = opensr_model.SRLatentDiffusion(
    config,
    device=device
)


# ============================================================
# 9. Load local pretrained checkpoint
# ============================================================

checkpoint_path = (
    Path(__file__).resolve().parent
    / "opensr-ldsrs2_v1_0_0.ckpt"
)

print(f"Loading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device)

weights = checkpoint["state_dict"]

# Remove loss-related parameters, matching the official loader
for key in list(weights.keys()):
    if "loss" in key:
        del weights[key]

model.model.load_state_dict(weights, strict=True)
model.eval()

print("Pretrained LDSR-S2 weights loaded successfully.")


# ============================================================
# 10. Run super-resolution
# ============================================================

print("\n🚀 Running LDSR-S2...")

with torch.no_grad():

    sr = model.forward(
        lr,
        sampling_steps=100
    )


print(f"SR output shape: {sr.shape}")


# ============================================================
# 11. Move result back to CPU
# ============================================================

sr = sr.squeeze(0).cpu().numpy()

print(f"Output NumPy shape: {sr.shape}")


# ============================================================
# 12. Save output as GeoTIFF
# ============================================================

output_path = OUTPUT_DIR / "ldsr_s2_real_test.tif"

out_profile = profile.copy()

out_profile.update(
    driver="GTiff",
    height=sr.shape[1],
    width=sr.shape[2],
    count=4,
    dtype="float32",
    transform=rasterio.Affine(
        profile["transform"].a / 4,
        profile["transform"].b,
        profile["transform"].c + col_start * profile["transform"].a,

        profile["transform"].d,
        profile["transform"].e / 4,
        profile["transform"].f + row_start * profile["transform"].e,
    ),
)

with rasterio.open(output_path, "w", **out_profile) as dst:

    dst.write(sr.astype(np.float32))


print(f"\nSaved SR image to:")
print(output_path)

print("\n🔥 REAL SENTINEL-2 → LDSR-S2 SUCCESS 🔥")