import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from scipy.ndimage import zoom


REPO = Path(__file__).resolve().parents[2]

PATCH_DIR = REPO / "data/raw/test_patches/patch_05_dense_urban/patch"

LDSR_PATH = (
    REPO
    / "data/outputs/benchmark/ldsr_s2/patch_05"
    / "ldsr_s2_patch_05.tif"
)

OUTPUT_PATH = (
    REPO
    / "data/outputs/benchmark/ldsr_s2/patch_05"
    / "comparison.png"
)


# ---------------------------------------------------------
# Load original RGB
# ---------------------------------------------------------

def load_band(name):
    with rasterio.open(PATCH_DIR / f"{name}.tif") as src:
        return src.read(1).astype(np.float32)


blue = load_band("B02")
green = load_band("B03")
red = load_band("B04")


# ---------------------------------------------------------
# Percentile stretch
# ---------------------------------------------------------

def stretch(x):
    valid = x > 0

    lo, hi = np.percentile(x[valid], [2, 98])

    return np.clip(
        (x - lo) / (hi - lo),
        0,
        1
    )


original_rgb = np.stack(
    [
        stretch(red),
        stretch(green),
        stretch(blue),
    ],
    axis=-1
)


# ---------------------------------------------------------
# Bicubic 4× baseline
# ---------------------------------------------------------

bicubic_rgb = zoom(
    original_rgb,
    (4, 4, 1),
    order=3
)


# ---------------------------------------------------------
# Load LDSR output
# ---------------------------------------------------------

with rasterio.open(LDSR_PATH) as src:
    ldsr = src.read().astype(np.float32)

# LDSR band order = B02, B03, B04, B08
ldsr_rgb = np.stack(
    [
        stretch(ldsr[2]),
        stretch(ldsr[1]),
        stretch(ldsr[0]),
    ],
    axis=-1
)


# ---------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

axes[0].imshow(original_rgb)
axes[0].set_title("Original Sentinel-2 (10m)")
axes[0].axis("off")

axes[1].imshow(bicubic_rgb)
axes[1].set_title("Bicubic Upsampling (2.5m)")
axes[1].axis("off")

axes[2].imshow(ldsr_rgb)
axes[2].set_title("LDSR-S2 (2.5m)")
axes[2].axis("off")

fig.suptitle(
    "Patch 05 — Dense Urban / Infrastructure",
    fontsize=14
)

plt.tight_layout()

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Comparison saved to:")
print(OUTPUT_PATH)