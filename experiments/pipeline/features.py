from pathlib import Path

import cv2
import numpy as np
import rasterio


EPS = 1e-6


# ============================================================
# BAND LOADING
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


def load_rgbn(patch_dir):

    bands = {}

    for band in [
        "B04",
        "B03",
        "B02",
        "B08",
    ]:

        path = find_band(
            Path(patch_dir),
            band,
        )

        with rasterio.open(path) as src:

            arr = src.read(
                1
            ).astype(
                np.float32
            )

        # Sentinel-2 L2A DN → reflectance
        arr /= 10000.0

        bands[band] = arr

    return bands


# ============================================================
# SAFE STATISTICS
# ============================================================

def mean(arr):
    return float(np.mean(arr))


def std(arr):
    return float(np.std(arr))


def fraction(condition):
    return float(np.mean(condition))


# ============================================================
# SCENE FEATURES
# ============================================================

def extract_features(bands):

    B02 = bands["B02"]
    B03 = bands["B03"]
    B04 = bands["B04"]
    B08 = bands["B08"]

    # --------------------------------------------------------
    # Vegetation index
    # --------------------------------------------------------

    ndvi = (
        B08 - B04
    ) / (
        B08 + B04 + EPS
    )

    # --------------------------------------------------------
    # Water index
    #
    # Using Green/NIR because we currently only have RGBN.
    # --------------------------------------------------------

    ndwi = (
        B03 - B08
    ) / (
        B03 + B08 + EPS
    )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = (
        B02
        + B03
        + B04
    ) / 3.0

    # --------------------------------------------------------
    # Spatial variance
    # --------------------------------------------------------

    spatial_variance = float(
        np.var(
            brightness
        )
    )

    # --------------------------------------------------------
    # Edge density
    # --------------------------------------------------------

    normalized = np.clip(
        brightness,
        0,
        1,
    )

    image_8bit = (
        normalized * 255
    ).astype(
        np.uint8
    )

    edges = cv2.Canny(
        image_8bit,
        50,
        150,
    )

    edge_density = float(
        np.mean(
            edges > 0
        )
    )

    # --------------------------------------------------------
    # Feature dictionary
    # --------------------------------------------------------

    features = {

        # Spectral statistics
        "B02_mean": mean(B02),
        "B02_std": std(B02),

        "B03_mean": mean(B03),
        "B03_std": std(B03),

        "B04_mean": mean(B04),
        "B04_std": std(B04),

        "B08_mean": mean(B08),
        "B08_std": std(B08),

        # Vegetation
        "ndvi_mean": mean(ndvi),
        "ndvi_std": std(ndvi),

        "vegetation_fraction": fraction(
            ndvi > 0.30
        ),

        # Water
        "ndwi_mean": mean(ndwi),
        "ndwi_std": std(ndwi),

        "water_fraction": fraction(
            ndwi > 0.20
        ),

        # Brightness
        "brightness_mean": mean(
            brightness
        ),

        "brightness_std": std(
            brightness
        ),

        # Spatial structure
        "spatial_variance": spatial_variance,

        "edge_density": edge_density,
    }

    return features


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def extract_scene_features(
    patch_dir
):

    bands = load_rgbn(
        patch_dir
    )

    return extract_features(
        bands
    )


# ============================================================
# PRETTY PRINT
# ============================================================

def print_features(features):

    print()
    print("=" * 70)
    print("SCENE FEATURES")
    print("=" * 70)

    for key, value in features.items():

        print(
            f"{key:25s}: {value:.6f}"
        )

    print()