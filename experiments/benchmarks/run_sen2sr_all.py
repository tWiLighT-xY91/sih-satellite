import sys
from pathlib import Path

import numpy as np
import rasterio


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "experiments/models/sen2sr"

# Make local SEN2SR predictor.py importable
sys.path.insert(0, str(MODEL_DIR))

from predictor import SEN2SRPredictor


# ============================================================
# Patch configuration
# ============================================================

PATCHES = {
    "patch_02": {
        "directory": PROJECT_ROOT / "data/raw/test_patches/patch_02_urban/patch",
        "output": (
            PROJECT_ROOT
            / "data/outputs/benchmark/sen2sr/patch_02"
            / "sen2sr_patch_02.tif"
        ),
    },

    "patch_03": {
        "directory": PROJECT_ROOT / "data/raw/test_patches/patch_03_agriculture/patch",
        "output": (
            PROJECT_ROOT
            / "data/outputs/benchmark/sen2sr/patch_03"
            / "sen2sr_patch_03.tif"
        ),
    },

    "patch_04": {
        "directory": PROJECT_ROOT / "data/raw/test_patches/patch_04_water/patch",
        "output": (
            PROJECT_ROOT
            / "data/outputs/benchmark/sen2sr/patch_04"
            / "sen2sr_patch_04.tif"
        ),
    },

    "patch_05": {
        "directory": PROJECT_ROOT / "data/raw/test_patches/patch_05_dense_urban/patch",
        "output": (
            PROJECT_ROOT
            / "data/outputs/benchmark/sen2sr/patch_05"
            / "sen2sr_patch_05.tif"
        ),
    },
}


# ============================================================
# SEN2SR band order
#
# IMPORTANT:
# SEN2SR expects:
# B04, B03, B02, B08
# ============================================================

BANDS = ["B04", "B03", "B02", "B08"]


# ============================================================
# Load model ONCE
# ============================================================

print("=" * 70)
print("SEN2SR MULTI-PATCH BENCHMARK")
print("=" * 70)

predictor = SEN2SRPredictor(str(MODEL_DIR))

print(f"\nDevice: {predictor.device}")
print(f"Scaling factor: {predictor.scaling_factor}")
print(f"Patch size: {predictor.patch_size}")


# ============================================================
# Process each patch
# ============================================================

for patch_id, cfg in PATCHES.items():

    print("\n")
    print("=" * 70)
    print(f"PROCESSING {patch_id.upper()}")
    print("=" * 70)

    patch_dir = cfg["directory"]
    output_path = cfg["output"]

    arrays = []
    profile = None

    # --------------------------------------------------------
    # Verify patch directory
    # --------------------------------------------------------

    if not patch_dir.exists():
        raise FileNotFoundError(
            f"Patch directory does not exist:\n{patch_dir}"
        )

    print(f"Patch directory: {patch_dir}")

    # --------------------------------------------------------
    # Read the existing 128x128 benchmark patch
    # --------------------------------------------------------

    for band in BANDS:

        # ----------------------------------------------------
        # Standard naming:
        # B02.tif, B03.tif, B04.tif, B08.tif
        # ----------------------------------------------------

        tif_path = patch_dir / f"{band}.tif"

        # ----------------------------------------------------
        # Patch 02 naming:
        # 02.tif, 03.tif, 04.tif, 08.tif
        # ----------------------------------------------------

        if not tif_path.exists():

            short_name = band[1:]  # B04 -> 04

            tif_path = patch_dir / f"{short_name}.tif"

        # ----------------------------------------------------
        # Make sure the band actually exists
        # ----------------------------------------------------

        if not tif_path.exists():

            raise FileNotFoundError(
                f"Could not find {band} for {patch_id}.\n"
                f"Expected either:\n"
                f"  {patch_dir / f'{band}.tif'}\n"
                f"or:\n"
                f"  {patch_dir / f'{band[1:]}.tif'}"
            )

        print(f"  {band}: {tif_path.name}")

        # ----------------------------------------------------
        # Read band
        # ----------------------------------------------------

        with rasterio.open(tif_path) as src:

            data = src.read(1).astype(np.float32)

            if profile is None:
                profile = src.profile.copy()

        arrays.append(data)

    # --------------------------------------------------------
    # Stack bands
    # --------------------------------------------------------

    image = np.stack(arrays, axis=0)

    print(f"\nInput shape: {image.shape}")

    # --------------------------------------------------------
    # Verify benchmark patch size
    # --------------------------------------------------------

    if image.shape != (4, 128, 128):

        raise ValueError(
            f"{patch_id} is not a 4 x 128 x 128 patch.\n"
            f"Got: {image.shape}"
        )

    # --------------------------------------------------------
    # Normalize Sentinel-2 DN
    # --------------------------------------------------------

    print(
        f"Raw range: "
        f"{image.min():.4f} → {image.max():.4f}"
    )

    if image.max() > 2.0:

        image = np.clip(
            image / 10000.0,
            0.0,
            1.0,
        )

    print(
        f"Normalized range: "
        f"{image.min():.4f} → {image.max():.4f}"
    )

    # --------------------------------------------------------
    # Run SEN2SR
    # --------------------------------------------------------

    print("\n🚀 Running SEN2SR...")

    sr = predictor.predict(image)

    print(
        f"Output shape: {sr.shape}"
    )

    print(
        f"Output dtype: {sr.dtype}"
    )

    print(
        f"Output range: "
        f"{sr.min():.6f} → {sr.max():.6f}"
    )

    # --------------------------------------------------------
    # Verify output dimensions
    # --------------------------------------------------------

    expected_shape = (
        4,
        128 * predictor.scaling_factor,
        128 * predictor.scaling_factor,
    )

    if sr.shape != expected_shape:

        raise ValueError(
            f"Unexpected SEN2SR output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got: {sr.shape}"
        )

    # --------------------------------------------------------
    # Update GeoTIFF transform
    #
    # 10 m → 2.5 m
    # --------------------------------------------------------

    transform = profile["transform"]

    new_transform = rasterio.Affine(
        transform.a / predictor.scaling_factor,
        transform.b,
        transform.c,

        transform.d,
        transform.e / predictor.scaling_factor,
        transform.f,
    )

    # --------------------------------------------------------
    # Output GeoTIFF profile
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        output_path,
        "w",
        **out_profile,
    ) as dst:

        dst.write(sr.astype(np.float32))

    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    print("\n✅ Saved:")
    print(output_path)

    print(
        f"Resolution: "
        f"{abs(transform.a) / predictor.scaling_factor:.2f} m"
    )


# ============================================================
# Complete
# ============================================================

print("\n")
print("=" * 70)
print("🔥 SEN2SR PATCHES 02–05 COMPLETE 🔥")
print("=" * 70)