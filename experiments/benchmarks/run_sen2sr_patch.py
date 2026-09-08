import sys
from pathlib import Path

import numpy as np
import rasterio


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "experiments/models/sen2sr"
PATCH_DIR = PROJECT_ROOT / "data/raw/test_patches/patch_01_forest"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/outputs/benchmark/sen2sr/patch_01"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "sen2sr_patch_01.tif"

sys.path.insert(0, str(MODEL_DIR))

from predictor import SEN2SRPredictor


# ============================================================
# 2. Sentinel-2 band files
# ============================================================

BAND_FILES = {
    "B02": next(PATCH_DIR.glob("*_B02_10m.jp2")),
    "B03": next(PATCH_DIR.glob("*_B03_10m.jp2")),
    "B04": next(PATCH_DIR.glob("*_B04_10m.jp2")),
    "B08": next(PATCH_DIR.glob("*_B08_10m.jp2")),
}


# ============================================================
# 3. Read Sentinel-2 bands
#
# IMPORTANT:
# SEN2SR expects B04, B03, B02, B08
# ============================================================

print("\nLoading Sentinel-2 bands...")

bands = []
profile = None

for band_name in ["B04", "B03", "B02", "B08"]:

    path = BAND_FILES[band_name]

    print(f"  {band_name}: {path.name}")

    with rasterio.open(path) as src:

        data = src.read(1)

        if profile is None:
            profile = src.profile.copy()

        bands.append(data.astype(np.float32))


image = np.stack(bands, axis=0)

print(f"\nFull image shape: {image.shape}")


# ============================================================
# 4. Find EXACT SAME 128x128 patch as LDSR
# ============================================================

PATCH_SIZE = 128
STEP = 256

print("\nSearching for the same valid 128 x 128 patch...")

height, width = image.shape[1:]

patch = None
row_start = None
col_start = None

for row in range(0, height - PATCH_SIZE + 1, STEP):

    for col in range(0, width - PATCH_SIZE + 1, STEP):

        candidate = image[
            :,
            row:row + PATCH_SIZE,
            col:col + PATCH_SIZE
        ]

        valid_mask = np.all(candidate > 0, axis=0)

        valid_fraction = valid_mask.mean()

        if valid_fraction >= 0.99:

            patch = candidate
            row_start = row
            col_start = col

            print(
                f"Found patch at "
                f"row={row_start}, "
                f"col={col_start}, "
                f"valid={valid_fraction:.2%}"
            )

            break

    if patch is not None:
        break


if patch is None:

    raise RuntimeError(
        "Could not find a sufficiently valid 128x128 patch."
    )


print(f"Patch shape: {patch.shape}")


# ============================================================
# 5. Normalize Sentinel-2 DN
# ============================================================

patch = patch / 10000.0

print("\nPatch statistics after /10000 normalization:")

for i, band_name in enumerate(
    ["B04", "B03", "B02", "B08"]
):

    band = patch[i]

    print(
        f"  {band_name}: "
        f"min={band.min():.4f}, "
        f"median={np.median(band):.4f}, "
        f"max={band.max():.4f}"
    )


# ============================================================
# 6. Load pretrained SEN2SR
# ============================================================

device = "cuda"

print("\nLoading SEN2SR...")

predictor = SEN2SRPredictor(
    str(MODEL_DIR)
)

print("Model loaded.")
print(f"Device: {predictor.device}")
print(f"Scaling factor: {predictor.scaling_factor}")


# ============================================================
# 7. Run super-resolution
# ============================================================

print("\n🚀 Running SEN2SR...")

sr = predictor.predict(patch)

print(f"\nSR output shape: {sr.shape}")
print(f"Output dtype: {sr.dtype}")
print(
    f"Output range: "
    f"{sr.min():.6f} → {sr.max():.6f}"
)


# ============================================================
# 8. Save output as GeoTIFF
# ============================================================

transform = profile["transform"]

new_transform = rasterio.Affine(
    transform.a / 4,
    transform.b,
    transform.c + col_start * transform.a,

    transform.d,
    transform.e / 4,
    transform.f + row_start * transform.e,
)

out_profile = profile.copy()

out_profile.update(
    driver="GTiff",
    height=sr.shape[1],
    width=sr.shape[2],
    count=4,
    dtype="float32",
    transform=new_transform,
    compress="lzw",
)

out_profile.pop("photometric", None)


with rasterio.open(
    OUTPUT_PATH,
    "w",
    **out_profile
) as dst:

    dst.write(sr.astype(np.float32))


print("\n🔥 SEN2SR SUCCESS 🔥")
print(f"Saved to: {OUTPUT_PATH}")
print("Output resolution: 2.5 m")