from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data/raw/test_patches/patch_01_forest"

LDSR_PATH = (
    PROJECT_ROOT
    / "data/outputs/ldsr_s2/ldsr_s2_real_test.tif"
)

SEN2SR_PATH = (
    PROJECT_ROOT
    / "data/outputs/benchmark/sen2sr/patch_01"
    / "sen2sr_patch_01.tif"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/outputs/benchmark"
    / "patch_01_ldsr_vs_sen2sr.png"
)


# ------------------------------------------------------------
# Read original Sentinel-2
# ------------------------------------------------------------

band_files = {
    "B02": next(RAW_DIR.glob("*_B02_10m.jp2")),
    "B03": next(RAW_DIR.glob("*_B03_10m.jp2")),
    "B04": next(RAW_DIR.glob("*_B04_10m.jp2")),
}


# Exact same crop used by the LDSR/SEN2SR benchmark
row_start = 0
col_start = 0
patch_size = 128

rgb = []

for band in ["B04", "B03", "B02"]:

    with rasterio.open(band_files[band]) as src:

        arr = src.read(
            1,
            window=rasterio.windows.Window(
                col_start,
                row_start,
                patch_size,
                patch_size,
            ),
        ).astype(np.float32)

    rgb.append(arr / 10000.0)

rgb = np.stack(rgb, axis=0)


# ------------------------------------------------------------
# Read SR outputs
# ------------------------------------------------------------

with rasterio.open(LDSR_PATH) as src:
    ldsr = src.read([1, 2, 3]).astype(np.float32)

with rasterio.open(SEN2SR_PATH) as src:
    sen2sr = src.read([1, 2, 3]).astype(np.float32)


# ------------------------------------------------------------
# Consistent visualization stretch
# ------------------------------------------------------------

# Use percentiles computed jointly across all images.
combined = np.concatenate(
    [
        rgb.ravel(),
        ldsr.ravel(),
        sen2sr.ravel(),
    ]
)

vmin = np.percentile(combined, 2)
vmax = np.percentile(combined, 98)


def normalize(x):
    return np.clip(
        (x - vmin) / (vmax - vmin),
        0,
        1,
    )


rgb_vis = normalize(rgb).transpose(1, 2, 0)
ldsr_vis = normalize(ldsr).transpose(1, 2, 0)
sen2sr_vis = normalize(sen2sr).transpose(1, 2, 0)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
)

axes[0].imshow(rgb_vis)
axes[0].set_title("Sentinel-2 — 10 m")

axes[1].imshow(ldsr_vis)
axes[1].set_title("LDSR-S2 — 2.5 m")

axes[2].imshow(sen2sr_vis)
axes[2].set_title("SEN2SR — 2.5 m")

for ax in axes:
    ax.axis("off")

fig.tight_layout()

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved comparison to:")
print(OUTPUT_PATH)