import rasterio
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "sentinel-2"

SR_PATH = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "ldsr_s2"
    / "ldsr_s2_real_test.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "ldsr_s2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Find Sentinel-2 bands
# ============================================================

band_paths = {
    "B02": next(DATA_DIR.glob("*_B02_10m.jp2")),
    "B03": next(DATA_DIR.glob("*_B03_10m.jp2")),
    "B04": next(DATA_DIR.glob("*_B04_10m.jp2")),
}


# ============================================================
# Read original patch
# ============================================================

patch_size = 128

bands = []

for band_name in ["B02", "B03", "B04"]:

    with rasterio.open(band_paths[band_name]) as src:
        data = src.read(
            1,
            window=rasterio.windows.Window(
                0,
                0,
                patch_size,
                patch_size
            )
        )

    bands.append(data.astype(np.float32) / 10000.0)


# RGB order: R=B04, G=B03, B=B02

original = np.stack(
    [bands[2], bands[1], bands[0]],
    axis=-1
)


# ============================================================
# Read LDSR output
# ============================================================

with rasterio.open(SR_PATH) as src:
    sr_data = src.read()

# LDSR output:
# B02 = index 0
# B03 = index 1
# B04 = index 2

sr = np.stack(
    [
        sr_data[2],
        sr_data[1],
        sr_data[0],
    ],
    axis=-1
)


# ============================================================
# Convert reflectance to displayable RGB
# ============================================================

def make_rgb(image):
    """
    Convert reflectance values to displayable RGB.

    Percentile stretching prevents the visualization
    from being dominated by extreme pixels.
    """

    image = np.nan_to_num(image)

    low = np.percentile(image, 2)
    high = np.percentile(image, 98)

    image = (image - low) / (high - low + 1e-8)

    image = np.clip(image, 0, 1)

    return image


original_rgb = make_rgb(original)
sr_rgb = make_rgb(sr)


# ============================================================
# Bicubic baseline
# ============================================================

original_uint8 = (
    original_rgb * 255
).astype(np.uint8)

bicubic = np.array(
    Image.fromarray(original_uint8).resize(
        (512, 512),
        Image.Resampling.BICUBIC
    )
) / 255.0


# ============================================================
# Create comparison figure
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

axes[0].imshow(original_rgb)
axes[0].set_title("Original Sentinel-2 (10m)")
axes[0].axis("off")

axes[1].imshow(bicubic)
axes[1].set_title("Bicubic Upsampling (2.5m)")
axes[1].axis("off")

axes[2].imshow(sr_rgb)
axes[2].set_title("LDSR-S2 (2.5m)")
axes[2].axis("off")

plt.tight_layout()

output_path = OUTPUT_DIR / "ldsr_s2_comparison.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Saved comparison to:")
print(output_path)