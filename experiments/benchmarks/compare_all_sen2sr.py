from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = PROJECT_ROOT / "data/raw/test_patches"

OUTPUT_ROOT = PROJECT_ROOT / "data/outputs/benchmark/comparisons"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Patch configuration
# ============================================================

PATCHES = {
    "patch_01": {
        "name": "Forest",
        "raw_dir": RAW_ROOT / "patch_01_forest",
        "ldsr": PROJECT_ROOT / "data/outputs/ldsr_s2/ldsr_s2_real_test.tif",
        "sen2sr": PROJECT_ROOT / "data/outputs/benchmark/sen2sr/patch_01/sen2sr_patch_01.tif",
        "raw_type": "jp2",
    },

    "patch_02": {
        "name": "Indian Peri-Urban",
        "raw_dir": RAW_ROOT / "patch_02_urban" / "patch",
        "ldsr": PROJECT_ROOT / "data/outputs/benchmark/ldsr_s2/patch_02/ldsr_s2_patch_02.tif",
        "sen2sr": PROJECT_ROOT / "data/outputs/benchmark/sen2sr/patch_02/sen2sr_patch_02.tif",
        "raw_type": "tif_02",
    },

    "patch_03": {
        "name": "Indian Agriculture",
        "raw_dir": RAW_ROOT / "patch_03_agriculture" / "patch",
        "ldsr": PROJECT_ROOT / "data/outputs/benchmark/ldsr_s2/patch_03/ldsr_s2_patch_03.tif",
        "sen2sr": PROJECT_ROOT / "data/outputs/benchmark/sen2sr/patch_03/sen2sr_patch_03.tif",
        "raw_type": "tif",
    },

    "patch_04": {
        "name": "Atmospheric / Cloud Stress",
        "raw_dir": RAW_ROOT / "patch_04_water" / "patch",
        "ldsr": PROJECT_ROOT / "data/outputs/benchmark/ldsr_s2/patch_04/ldsr_s2_patch_04.tif",
        "sen2sr": PROJECT_ROOT / "data/outputs/benchmark/sen2sr/patch_04/sen2sr_patch_04.tif",
        "raw_type": "tif",
    },

    "patch_05": {
        "name": "Dense Urban / Infrastructure",
        "raw_dir": RAW_ROOT / "patch_05_dense_urban" / "patch",
        "ldsr": PROJECT_ROOT / "data/outputs/benchmark/ldsr_s2/patch_05/ldsr_s2_patch_05.tif",
        "sen2sr": PROJECT_ROOT / "data/outputs/benchmark/sen2sr/patch_05/sen2sr_patch_05.tif",
        "raw_type": "tif",
    },
}


# ============================================================
# Read RGB input
# ============================================================

def read_rgb(cfg):

    raw_dir = cfg["raw_dir"]

    # --------------------------------------------------------
    # Patch 01: original JP2 files, exact crop = row 0, col 0
    # --------------------------------------------------------

    if cfg["raw_type"] == "jp2":

        bands = []

        for band in ["B04", "B03", "B02"]:

            path = next(raw_dir.glob(f"*_{band}_10m.jp2"))

            with rasterio.open(path) as src:

                arr = src.read(
                    1,
                    window=rasterio.windows.Window(
                        0,
                        0,
                        128,
                        128,
                    ),
                ).astype(np.float32)

            bands.append(arr / 10000.0)

        return np.stack(bands, axis=0)


    # --------------------------------------------------------
    # Patch 02: 04.tif, 03.tif, 02.tif
    # --------------------------------------------------------

    if cfg["raw_type"] == "tif_02":

        files = {
            "B04": raw_dir / "04.tif",
            "B03": raw_dir / "03.tif",
            "B02": raw_dir / "02.tif",
        }


    # --------------------------------------------------------
    # Patch 03-05: B04.tif, B03.tif, B02.tif
    # --------------------------------------------------------

    else:

        files = {
            "B04": raw_dir / "B04.tif",
            "B03": raw_dir / "B03.tif",
            "B02": raw_dir / "B02.tif",
        }


    bands = []

    for band in ["B04", "B03", "B02"]:

        with rasterio.open(files[band]) as src:

            arr = src.read(1).astype(np.float32)

        if arr.max() > 2.0:
            arr = arr / 10000.0

        bands.append(arr)

    return np.stack(bands, axis=0)


# ============================================================
# Consistent visualization
# ============================================================

def make_rgb(arr):

    # Joint percentile stretch.
    # This ensures all three images use the same scale.
    values = arr[np.isfinite(arr)]

    low = np.percentile(values, 2)
    high = np.percentile(values, 98)

    return np.clip(
        (arr - low) / (high - low),
        0,
        1,
    ).transpose(1, 2, 0)


# ============================================================
# Compare every patch
# ============================================================

for patch_id, cfg in PATCHES.items():

    print("\n" + "=" * 70)
    print(f"COMPARING {patch_id.upper()} — {cfg['name']}")
    print("=" * 70)

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    original = read_rgb(cfg)

    # --------------------------------------------------------
    # LDSR
    # --------------------------------------------------------

    with rasterio.open(cfg["ldsr"]) as src:

        ldsr = src.read([1, 2, 3]).astype(np.float32)

    # --------------------------------------------------------
    # SEN2SR
    # --------------------------------------------------------

    with rasterio.open(cfg["sen2sr"]) as src:

        sen2sr = src.read([1, 2, 3]).astype(np.float32)

    print("Original:", original.shape)
    print("LDSR:", ldsr.shape)
    print("SEN2SR:", sen2sr.shape)

    # --------------------------------------------------------
    # Shared visualization range
    # --------------------------------------------------------

    combined = np.concatenate(
        [
            original.ravel(),
            ldsr.ravel(),
            sen2sr.ravel(),
        ]
    )

    low = np.percentile(combined, 2)
    high = np.percentile(combined, 98)

    def normalize(x):

        return np.clip(
            (x - low) / (high - low),
            0,
            1,
        ).transpose(1, 2, 0)

    original_rgb = normalize(original)
    ldsr_rgb = normalize(ldsr)
    sen2sr_rgb = normalize(sen2sr)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(original_rgb)
    axes[0].set_title("Sentinel-2 — 10 m")

    axes[1].imshow(ldsr_rgb)
    axes[1].set_title("LDSR-S2 — 2.5 m")

    axes[2].imshow(sen2sr_rgb)
    axes[2].set_title("SEN2SR — 2.5 m")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{patch_id.upper()} — {cfg['name']}",
        fontsize=14,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / f"{patch_id}_ldsr_vs_sen2sr.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


print("\n" + "=" * 70)
print("🔥 ALL LDSR vs SEN2SR COMPARISONS COMPLETE 🔥")
print("=" * 70)