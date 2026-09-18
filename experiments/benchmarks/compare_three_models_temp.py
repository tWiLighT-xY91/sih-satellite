from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import Window


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "sih" / "sih-satellite"

RAW_ROOT = (
    ROOT / "data/raw/test_patches"
)

OUTPUT_ROOT = (
    ROOT
    / "data/outputs/benchmark/three_model_comparison"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PATCHES
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
# FIND BAND
# ============================================================

def find_band(patch_dir, band):

    short = band.replace("B", "")

    candidates = [
        patch_dir / f"{band}.tif",
        patch_dir / f"{short}.tif",
        patch_dir / f"{band}_10m.tif",
        patch_dir / f"{short}_10m.tif",
    ]

    for path in candidates:

        if path.exists():
            return path

    jp2_files = list(
        patch_dir.glob(
            f"*{band}_10m.jp2"
        )
    )

    if jp2_files:
        return jp2_files[0]

    raise FileNotFoundError(
        f"Could not find {band} in {patch_dir}"
    )


# ============================================================
# LOAD ORIGINAL SENTINEL-2 RGB
#
# RGB = B04, B03, B02
# ============================================================

def load_original(config):

    arrays = []

    for band in [
        "B04",
        "B03",
        "B02",
    ]:

        path = find_band(
            config["dir"],
            band,
        )

        with rasterio.open(path) as src:

            if (
                src.width == 128
                and src.height == 128
            ):

                arr = src.read(
                    1
                ).astype(
                    np.float32
                )

            else:

                arr = src.read(
                    1,
                    window=Window(
                        config["col"],
                        config["row"],
                        128,
                        128,
                    ),
                ).astype(
                    np.float32
                )

        # Sentinel-2 L2A DN → reflectance
        arr /= 10000.0

        arrays.append(arr)

    return np.stack(
        arrays,
        axis=0,
    )


# ============================================================
# LOAD MODEL OUTPUT
# ============================================================

def load_model_output(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing model output:\n{path}"
        )

    with rasterio.open(path) as src:

        return src.read().astype(
            np.float32
        )


# ============================================================
# MODEL OUTPUT PATH
# ============================================================

def model_path(
    patch_id,
    model,
):

    return (
        ROOT
        / "data/outputs/benchmark"
        / model
        / patch_id
        / f"{model}_{patch_id}.tif"
    )


# ============================================================
# RGB CONVERSION
# ============================================================

def get_rgb(
    model,
    arr,
):

    if model == "ldsr_s2":

        # LDSR output:
        # B02, B03, B04, B08
        return arr[
            [2, 1, 0]
        ]

    elif model in [
        "sen2sr",
        "mamba",
    ]:

        # SEN2SR / Mamba:
        # B04, B03, B02, B08
        return arr[
            [0, 1, 2]
        ]

    else:

        raise ValueError(
            f"Unknown model: {model}"
        )


# ============================================================
# SHARED 2–98 PERCENTILE STRETCH
#
# EXACTLY like our original comparison.
#
# The stretch is calculated jointly across:
#
# Original + LDSR + SEN2SR + Mamba
#
# for EACH patch.
# ============================================================

def stretch_images(images):

    combined = np.concatenate(
        [
            image.reshape(
                3,
                -1,
            )
            for image in images
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

    # Prevent division by zero
    high = np.maximum(
        high,
        low + 1e-6,
    )

    stretched = []

    for image in images:

        result = (
            image
            - low[:, None, None]
        ) / (
            high - low
        )[:, None, None]

        result = np.clip(
            result,
            0,
            1,
        )

        # CHW → HWC
        result = np.transpose(
            result,
            (1, 2, 0),
        )

        stretched.append(
            result
        )

    return stretched


# ============================================================
# UPSCALE ORIGINAL ONLY FOR DISPLAY
# ============================================================

def upscale_original(image):

    return np.repeat(
        np.repeat(
            image,
            4,
            axis=1,
        ),
        4,
        axis=2,
    )


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 80)
print(
    "THREE-MODEL REAL-SCENE COMPARISON"
)
print(
    "LDSR-S2 vs SEN2SR CNN vs Mamba-SEN2SR"
)
print("=" * 80)


for patch_id, config in PATCHES.items():

    print()
    print("-" * 80)
    print(
        config["name"]
    )
    print("-" * 80)

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    original = load_original(
        config
    )

    # --------------------------------------------------------
    # Load LDSR
    # --------------------------------------------------------

    ldsr = load_model_output(
        model_path(
            patch_id,
            "ldsr_s2",
        )
    )

    # --------------------------------------------------------
    # Load SEN2SR CNN
    # --------------------------------------------------------

    sen2sr = load_model_output(
        model_path(
            patch_id,
            "sen2sr",
        )
    )

    # --------------------------------------------------------
    # Load Mamba
    # --------------------------------------------------------

    mamba_path = (
        ROOT
        / "data/outputs/benchmark"
        / "mamba"
        / patch_id
        / f"mamba_{patch_id}.tif"
    )

    mamba = load_model_output(
        mamba_path
    )

    # --------------------------------------------------------
    # Convert everything to RGB
    # --------------------------------------------------------

    original_rgb = original

    ldsr_rgb = get_rgb(
        "ldsr_s2",
        ldsr,
    )

    sen2sr_rgb = get_rgb(
        "sen2sr",
        sen2sr,
    )

    mamba_rgb = get_rgb(
        "mamba",
        mamba,
    )

    images = [
        original_rgb,
        ldsr_rgb,
        sen2sr_rgb,
        mamba_rgb,
    ]

    # --------------------------------------------------------
    # Shared stretch
    # --------------------------------------------------------

    display_images = stretch_images(
        images
    )

    # Original is 128×128.
    # Upscale ONLY for visual consistency.
    display_images[0] = (
        upscale_original(
            np.transpose(
                display_images[0],
                (2, 0, 1),
            )
        )
    )

    # Convert back to HWC
    display_images[0] = np.transpose(
        display_images[0],
        (1, 2, 0),
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4.5),
    )

    titles = [
        "Original Sentinel-2\n10 m",
        "LDSR-S2\n2.5 m",
        "SEN2SR CNN\n2.5 m",
        "Mamba-SEN2SR\n2.5 m",
    ]

    for ax, image, title in zip(
        axes,
        display_images,
        titles,
    ):

        ax.imshow(
            image
        )

        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
        )

        ax.axis("off")

    # --------------------------------------------------------
    # Figure title
    # --------------------------------------------------------

    fig.suptitle(
        config["name"],
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save individual patch image
    # --------------------------------------------------------

    output_path = (
        OUTPUT_ROOT
        / f"{patch_id}_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        output_path,
    )


print()
print("=" * 80)
print(
    "ALL FIVE PATCH COMPARISONS COMPLETE"
)
print("=" * 80)

print()
print(
    "Output directory:"
)

print(
    OUTPUT_ROOT
)