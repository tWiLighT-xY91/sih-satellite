from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import Window


# ============================================================
# Paths
# ============================================================

ROOT = Path.home() / "sih" / "sih-satellite"

RAW_ROOT = (
    ROOT / "data/raw/test_patches"
)

OUTPUT_ROOT = (
    ROOT / "data/outputs/benchmark"
)

OUTPUT_PATH = (
    OUTPUT_ROOT
    / "all_five_patches_four_model_comparison.png"
)


# ============================================================
# Patch configuration
# ============================================================

PATCHES = {

    "patch_01": {
        "name": "Patch 01 — Forest",
        "dir": RAW_ROOT / "patch_01_forest/patch",
        "row": 0,
        "col": 0,
    },

    "patch_02": {
        "name": "Patch 02 — Peri-urban India",
        "dir": RAW_ROOT / "patch_02_urban/patch",
        "row": 6144,
        "col": 5632,
    },

    "patch_03": {
        "name": "Patch 03 — Agriculture India",
        "dir": RAW_ROOT / "patch_03_agriculture/patch",
        "row": 5632,
        "col": 5120,
    },

    "patch_04": {
        "name": "Patch 04 — Atmospheric Stress",
        "dir": RAW_ROOT / "patch_04_atmospheric/patch",
        "row": 2048,
        "col": 5632,
    },

    "patch_05": {
        "name": "Patch 05 — Dense Urban",
        "dir": RAW_ROOT / "patch_05_urban_dense/patch",
        "row": 7168,
        "col": 6656,
    },
}


# ============================================================
# Model output paths
# ============================================================

def model_path(model, patch):

    return (
        OUTPUT_ROOT
        / "benchmark"
        / model
        / patch
        / f"{model}_{patch}.tif"
    )


# ============================================================
# Correct output paths
# ============================================================

MODEL_PATHS = {

    "patch_01": {
        "ldsr": (
            OUTPUT_ROOT
            / "ldsr_s2/patch_01"
            / "ldsr_s2_patch_01.tif"
        ),
        "sen2sr": (
            OUTPUT_ROOT
            / "sen2sr/patch_01"
            / "sen2sr_patch_01.tif"
        ),
        "satlas": (
            OUTPUT_ROOT
            / "satlas/patch_01"
            / "satlas_patch_01.tif"
        ),
    },

    "patch_02": {
        "ldsr": (
            OUTPUT_ROOT
            / "ldsr_s2/patch_02"
            / "ldsr_s2_patch_02.tif"
        ),
        "sen2sr": (
            OUTPUT_ROOT
            / "sen2sr/patch_02"
            / "sen2sr_patch_02.tif"
        ),
        "satlas": (
            OUTPUT_ROOT
            / "satlas/patch_02"
            / "satlas_patch_02.tif"
        ),
    },

    "patch_03": {
        "ldsr": (
            OUTPUT_ROOT
            / "ldsr_s2/patch_03"
            / "ldsr_s2_patch_03.tif"
        ),
        "sen2sr": (
            OUTPUT_ROOT
            / "sen2sr/patch_03"
            / "sen2sr_patch_03.tif"
        ),
        "satlas": (
            OUTPUT_ROOT
            / "satlas/patch_03"
            / "satlas_patch_03.tif"
        ),
    },

    "patch_04": {
        "ldsr": (
            OUTPUT_ROOT
            / "ldsr_s2/patch_04"
            / "ldsr_s2_patch_04.tif"
        ),
        "sen2sr": (
            OUTPUT_ROOT
            / "sen2sr/patch_04"
            / "sen2sr_patch_04.tif"
        ),
        "satlas": (
            OUTPUT_ROOT
            / "satlas/patch_04"
            / "satlas_patch_04.tif"
        ),
    },

    "patch_05": {
        "ldsr": (
            OUTPUT_ROOT
            / "ldsr_s2/patch_05"
            / "ldsr_s2_patch_05.tif"
        ),
        "sen2sr": (
            OUTPUT_ROOT
            / "sen2sr/patch_05"
            / "sen2sr_patch_05.tif"
        ),
        "satlas": (
            OUTPUT_ROOT
            / "satlas/patch_05"
            / "satlas_patch_05.tif"
        ),
    },
}


# ============================================================
# Find band
#
# Supports all naming conventions currently present
# in our benchmark patches.
# ============================================================

def find_band(patch_dir, band):

    short_name = band.replace("B", "")

    candidates = [

        patch_dir / f"{band}.tif",
        patch_dir / f"{short_name}.tif",

        patch_dir / f"{band}_10m.tif",
        patch_dir / f"{short_name}_10m.tif",
    ]

    for path in candidates:

        if path.exists():
            return path

    # Original Sentinel-2 JP2 fallback
    files = list(
        patch_dir.glob(
            f"*{band}_10m.jp2"
        )
    )

    if files:
        return files[0]

    available = [
        p.name
        for p in patch_dir.iterdir()
    ]

    raise FileNotFoundError(
        f"\nCould not find {band} in {patch_dir}\n"
        f"Available files:\n"
        + "\n".join(
            f"  - {name}"
            for name in available
        )
    )


# ============================================================
# Load original Sentinel-2 RGB
#
# RGB:
# B04 = Red
# B03 = Green
# B02 = Blue
# ============================================================

def load_original(patch_config):

    patch_dir = patch_config["dir"]

    row = patch_config["row"]
    col = patch_config["col"]

    arrays = []

    for band in [
        "B04",
        "B03",
        "B02",
    ]:

        path = find_band(
            patch_dir,
            band,
        )

        with rasterio.open(path) as src:

            # Already-extracted 128×128 benchmark patch
            if (
                src.width == 128
                and src.height == 128
            ):

                arr = src.read(
                    1
                ).astype(
                    np.float32
                )

            # Original Sentinel-2 JP2
            else:

                window = Window(
                    col,
                    row,
                    128,
                    128,
                )

                arr = src.read(
                    1,
                    window=window,
                ).astype(
                    np.float32
                )

        # Sentinel-2 L2A DN → reflectance
        arr /= 10000.0

        arrays.append(arr)

    rgb = np.stack(
        arrays,
        axis=0,
    )

    return np.clip(
        rgb,
        0,
        1,
    )


# ============================================================
# Load SR output
# ============================================================

def load_sr(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing model output:\n{path}"
        )

    with rasterio.open(path) as src:

        arr = src.read().astype(
            np.float32
        )

    return np.clip(
        arr,
        0,
        1,
    )


# ============================================================
# Convert model output to RGB
# ============================================================

def get_rgb_images(
    original,
    ldsr,
    sen2sr,
    satlas,
):

    # Original is already:
    # B04, B03, B02
    original_rgb = original

    # LDSR runner used:
    # B02, B03, B04, B08
    #
    # RGB = B04, B03, B02
    ldsr_rgb = ldsr[
        [2, 1, 0]
    ]

    # SEN2SR main model:
    # B04, B03, B02, B08
    sen2sr_rgb = sen2sr[
        [0, 1, 2]
    ]

    # Satlas output is already RGB
    satlas_rgb = satlas

    return [
        original_rgb,
        ldsr_rgb,
        sen2sr_rgb,
        satlas_rgb,
    ]


# ============================================================
# Shared percentile stretch
#
# IMPORTANT:
# Stretch is calculated separately for each patch,
# but jointly across all four models.
# ============================================================

def stretch_images(images):

    combined = np.concatenate(
        [
            img.reshape(
                3,
                -1
            )
            for img in images
        ],
        axis=1,
    )

    low = np.percentile(
        combined,
        2,
        axis=1,
    )

    high = np.percentile(
        combined,
        98,
        axis=1,
    )

    stretched = []

    for img in images:

        result = (
            img
            - low[:, None, None]
        ) / (
            high - low
        )[:, None, None]

        result = np.clip(
            result,
            0,
            1,
        )

        stretched.append(
            np.transpose(
                result,
                (1, 2, 0),
            )
        )

    return stretched


# ============================================================
# Upscale original ONLY for display
#
# This does NOT modify the original data.
# ============================================================

def upscale_original(image):

    return np.repeat(
        np.repeat(
            image,
            4,
            axis=0,
        ),
        4,
        axis=1,
    )


# ============================================================
# Main
# ============================================================

print()
print("=" * 70)
print("FOUR-MODEL COMPARISON — ALL FIVE PATCHES")
print("=" * 70)


results = {}


for patch_id, patch_config in PATCHES.items():

    print()
    print("-" * 70)
    print(
        patch_config["name"]
    )
    print("-" * 70)

    # --------------------------------------------------------
    # Load original
    # --------------------------------------------------------

    original = load_original(
        patch_config
    )

    # --------------------------------------------------------
    # Load model outputs
    # --------------------------------------------------------

    ldsr = load_sr(
        MODEL_PATHS[patch_id]["ldsr"]
    )

    sen2sr = load_sr(
        MODEL_PATHS[patch_id]["sen2sr"]
    )

    satlas = load_sr(
        MODEL_PATHS[patch_id]["satlas"]
    )

    print(
        "Original:",
        original.shape,
    )

    print(
        "LDSR:",
        ldsr.shape,
    )

    print(
        "SEN2SR:",
        sen2sr.shape,
    )

    print(
        "Satlas:",
        satlas.shape,
    )

    # --------------------------------------------------------
    # RGB conversion
    # --------------------------------------------------------

    images = get_rgb_images(
        original,
        ldsr,
        sen2sr,
        satlas,
    )

    # --------------------------------------------------------
    # Shared stretch
    # --------------------------------------------------------

    display_images = stretch_images(
        images
    )

    # Upscale original only for visual consistency
    display_images[0] = upscale_original(
        display_images[0]
    )

    results[patch_id] = display_images


# ============================================================
# Create 5 × 4 comparison figure
# ============================================================

fig, axes = plt.subplots(
    5,
    4,
    figsize=(16, 20),
)


column_titles = [
    "Original Sentinel-2",
    "LDSR",
    "SEN2SR",
    "Satlas ESRGAN",
]


for col, title in enumerate(
    column_titles
):

    axes[0, col].set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=12,
    )


for row, (patch_id, patch_config) in enumerate(
    PATCHES.items()
):

    images = results[patch_id]

    for col, image in enumerate(
        images
    ):

        ax = axes[row, col]

        ax.imshow(image)

        ax.axis("off")

    # Patch label on left
    axes[row, 0].set_ylabel(
        patch_config["name"],
        fontsize=12,
        fontweight="bold",
        rotation=90,
        labelpad=15,
    )


# ============================================================
# Figure title
# ============================================================

fig.suptitle(
    "Sentinel-2 Super-Resolution Model Comparison\n"
    "Real-Scene Benchmark — Five Test Patches",
    fontsize=18,
    fontweight="bold",
    y=0.995,
)


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.98,
    ]
)


# ============================================================
# Save
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)


print()
print("=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)

print()
print("Saved to:")
print(OUTPUT_PATH)
print()
print(
    "Each patch uses a shared 2nd–98th percentile "
    "stretch across all four models."
)
print()
print(
    "Original Sentinel-2 is upscaled ONLY for display."
)
print()