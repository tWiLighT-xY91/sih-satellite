from pathlib import Path
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import Window


# ============================================================
# Mamba model
# ============================================================

ROOT = Path.home() / "sih" / "sih-satellite"

MAMBA_DIR = (
    ROOT
    / "checkpoints"
    / "sen2sr-mamba-rgbn-x4"
).resolve()

sys.path.insert(0, str(MAMBA_DIR))

from model import Model as MambaModel


# ============================================================
# Paths
# ============================================================

RAW_ROOT = ROOT / "data/raw/test_patches"

OUTPUT_ROOT = ROOT / "data/outputs/benchmark"

OUTPUT_PATH = (
    OUTPUT_ROOT
    / "all_five_patches_four_model_comparison.png"
)

MAMBA_OUTPUT_ROOT = OUTPUT_ROOT / "mamba"


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
# Existing model output paths
# ============================================================

MODEL_PATHS = {

    "patch_01": {
        "ldsr": OUTPUT_ROOT
        / "ldsr_s2/patch_01"
        / "ldsr_s2_patch_01.tif",

        "sen2sr": OUTPUT_ROOT
        / "sen2sr/patch_01"
        / "sen2sr_patch_01.tif",

        "satlas": OUTPUT_ROOT
        / "satlas/patch_01"
        / "satlas_patch_01.tif",
    },

    "patch_02": {
        "ldsr": OUTPUT_ROOT
        / "ldsr_s2/patch_02"
        / "ldsr_s2_patch_02.tif",

        "sen2sr": OUTPUT_ROOT
        / "sen2sr/patch_02"
        / "sen2sr_patch_02.tif",

        "satlas": OUTPUT_ROOT
        / "satlas/patch_02"
        / "satlas_patch_02.tif",
    },

    "patch_03": {
        "ldsr": OUTPUT_ROOT
        / "ldsr_s2/patch_03"
        / "ldsr_s2_patch_03.tif",

        "sen2sr": OUTPUT_ROOT
        / "sen2sr/patch_03"
        / "sen2sr_patch_03.tif",

        "satlas": OUTPUT_ROOT
        / "satlas/patch_03"
        / "satlas_patch_03.tif",
    },

    "patch_04": {
        "ldsr": OUTPUT_ROOT
        / "ldsr_s2/patch_04"
        / "ldsr_s2_patch_04.tif",

        "sen2sr": OUTPUT_ROOT
        / "sen2sr/patch_04"
        / "sen2sr_patch_04.tif",

        "satlas": OUTPUT_ROOT
        / "satlas/patch_04"
        / "satlas_patch_04.tif",
    },

    "patch_05": {
        "ldsr": OUTPUT_ROOT
        / "ldsr_s2/patch_05"
        / "ldsr_s2_patch_05.tif",

        "sen2sr": OUTPUT_ROOT
        / "sen2sr/patch_05"
        / "sen2sr_patch_05.tif",

        "satlas": OUTPUT_ROOT
        / "satlas/patch_05"
        / "satlas_patch_05.tif",
    },
}


# ============================================================
# Find band
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
# Load 10m band
# ============================================================

def load_band(patch_config, band):

    patch_dir = patch_config["dir"]

    row = patch_config["row"]
    col = patch_config["col"]

    path = find_band(
        patch_dir,
        band,
    )

    with rasterio.open(path) as src:

        if (
            src.width == 128
            and src.height == 128
        ):

            arr = src.read(1).astype(
                np.float32
            )

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

    return arr / 10000.0


# ============================================================
# Load original RGB
# ============================================================

def load_original(patch_config):

    arrays = []

    for band in [
        "B04",
        "B03",
        "B02",
    ]:

        arrays.append(
            load_band(
                patch_config,
                band,
            )
        )

    return np.clip(
        np.stack(
            arrays,
            axis=0,
        ),
        0,
        1,
    )


# ============================================================
# Load RGBN for Mamba
#
# OFFICIAL ORDER:
# B04, B03, B02, B08
# ============================================================

def load_mamba_input(patch_config):

    arrays = []

    for band in [
        "B04",
        "B03",
        "B02",
        "B08",
    ]:

        arrays.append(
            load_band(
                patch_config,
                band,
            )
        )

    return np.stack(
        arrays,
        axis=0,
    ).astype(
        np.float32
    )


# ============================================================
# Load existing SR output
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
# Save Mamba output
# ============================================================

def save_mamba_output(
    output,
    patch_config,
    output_path,
):

    source_path = find_band(
        patch_config["dir"],
        "B04",
    )

    with rasterio.open(
        source_path
    ) as src:

        profile = src.profile.copy()

        transform = src.transform
        crs = src.crs

        if not (
            src.width == 128
            and src.height == 128
        ):

            transform = src.window_transform(
                Window(
                    patch_config["col"],
                    patch_config["row"],
                    128,
                    128,
                )
            )

    # 4x spatial resolution
    transform = transform * transform.scale(
        1 / 4,
        1 / 4,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile.update(
        driver="GTiff",
        height=output.shape[1],
        width=output.shape[2],
        count=4,
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="deflate",
    )

    # Clip ONLY for saved real-scene reflectance output.
    output = np.clip(
        output,
        0,
        1,
    ).astype(
        np.float32
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(output)


# ============================================================
# Run Mamba
# ============================================================

def run_mamba(
    mamba_model,
    patch_id,
    patch_config,
):

    output_path = (
        MAMBA_OUTPUT_ROOT
        / patch_id
        / f"mamba_{patch_id}.tif"
    )

    if output_path.exists():

        print(
            "Mamba output already exists:"
        )
        print(
            output_path
        )

        return load_sr(
            output_path
        )

    print(
        "\nRunning Mamba-SEN2SR..."
    )

    image = load_mamba_input(
        patch_config
    )

    print(
        "Mamba input:",
        image.shape,
        "range:",
        float(image.min()),
        float(image.max()),
    )

    output = mamba_model.predict(
        image
    )

    output = np.asarray(
        output,
        dtype=np.float32,
    )

    print(
        "Mamba output:",
        output.shape,
        "raw range:",
        float(output.min()),
        float(output.max()),
    )

    save_mamba_output(
        output,
        patch_config,
        output_path,
    )

    print(
        "Saved Mamba:",
        output_path,
    )

    return np.clip(
        output,
        0,
        1,
    )


# ============================================================
# Convert all model outputs to RGB
# ============================================================

def get_rgb_images(
    original,
    ldsr,
    sen2sr,
    satlas,
    mamba,
):

    # Original:
    # B04, B03, B02
    original_rgb = original

    # LDSR:
    # B02, B03, B04, B08
    ldsr_rgb = ldsr[
        [2, 1, 0]
    ]

    # SEN2SR CNN:
    # B04, B03, B02, B08
    sen2sr_rgb = sen2sr[
        [0, 1, 2]
    ]

    # Satlas:
    # RGB
    satlas_rgb = satlas

    # Mamba:
    # B04, B03, B02, B08
    mamba_rgb = mamba[
        [0, 1, 2]
    ]

    return [
        original_rgb,
        ldsr_rgb,
        sen2sr_rgb,
        satlas_rgb,
        mamba_rgb,
    ]


# ============================================================
# Shared percentile stretch
# ============================================================

def stretch_images(images):

    combined = np.concatenate(
        [
            img.reshape(
                3,
                -1,
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

    # Avoid division by zero
    high = np.maximum(
        high,
        low + 1e-6,
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
print("=" * 80)
print("FIVE-PATCH / FOUR-MODEL SR COMPARISON")
print("=" * 80)

print()
print("Loading Mamba-SEN2SR...")

mamba_model = MambaModel(
    local_dir=str(
        MAMBA_DIR
    )
)

print(
    "Mamba:",
    mamba_model.description,
)


results = {}


for patch_id, patch_config in PATCHES.items():

    print()
    print("-" * 80)
    print(
        patch_config["name"]
    )
    print("-" * 80)

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    original = load_original(
        patch_config
    )

    # --------------------------------------------------------
    # Existing models
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

    # --------------------------------------------------------
    # Mamba
    # --------------------------------------------------------

    mamba = run_mamba(
        mamba_model,
        patch_id,
        patch_config,
    )

    print()
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

    print(
        "Mamba:",
        mamba.shape,
    )

    # --------------------------------------------------------
    # RGB conversion
    # --------------------------------------------------------

    images = get_rgb_images(
        original,
        ldsr,
        sen2sr,
        satlas,
        mamba,
    )

    # --------------------------------------------------------
    # Shared stretch
    # --------------------------------------------------------

    display_images = stretch_images(
        images
    )

    # Original only upscaled for visual comparison
    display_images[0] = (
        upscale_original(
            display_images[0]
        )
    )

    results[patch_id] = (
        display_images
    )


# ============================================================
# Create 5 × 5 comparison figure
# ============================================================

fig, axes = plt.subplots(
    5,
    5,
    figsize=(20, 20),
)


column_titles = [
    "Original Sentinel-2",
    "LDSR-S2",
    "SEN2SR CNN",
    "Satlas ESRGAN",
    "Mamba-SEN2SR",
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


for row, (
    patch_id,
    patch_config,
) in enumerate(
    PATCHES.items()
):

    images = results[
        patch_id
    ]

    for col, image in enumerate(
        images
    ):

        ax = axes[
            row,
            col,
        ]

        ax.imshow(
            image
        )

        ax.axis(
            "off"
        )

    axes[
        row,
        0
    ].set_ylabel(
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


plt.close(fig)


print()
print("=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)

print()
print("Saved combined comparison:")
print(
    OUTPUT_PATH
)

print()
print(
    "Mamba outputs:"
)

for patch_id in PATCHES:

    print(
        MAMBA_OUTPUT_ROOT
        / patch_id
        / f"mamba_{patch_id}.tif"
    )

print()
print(
    "Each patch uses a shared 2nd–98th "
    "percentile stretch across all displayed models."
)

print(
    "Original Sentinel-2 is upscaled ONLY for display."
)

print()