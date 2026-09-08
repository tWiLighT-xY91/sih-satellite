from pathlib import Path
import sys

import numpy as np
import rasterio
import torch
from rasterio.windows import Window


# ============================================================
# Paths
# ============================================================

ROOT = Path.home() / "sih" / "sih-satellite"

SATLAS_ROOT = Path.home() / "sih" / "satlas-super-resolution"

CHECKPOINT = (
    SATLAS_ROOT / "esrgan_1S2.pth"
)

# Allow import of standalone Satlas model
sys.path.insert(
    0,
    str(SATLAS_ROOT / "ssr")
)

from satlas_rrdb_standalone import SSR_RRDBNet


# ============================================================
# Patch configuration
# ============================================================

PATCHES = {

    "patch_02": {
        "dir": ROOT / "data/raw/test_patches/patch_02_urban/patch",
        "row": 6144,
        "col": 5632,
    },

    "patch_03": {
        "dir": ROOT / "data/raw/test_patches/patch_03_agriculture/patch",
        "row": 5632,
        "col": 5120,
    },

    "patch_04": {
        "dir": ROOT / "data/raw/test_patches/patch_04_atmospheric/patch",
        "row": 2048,
        "col": 5632,
    },

    "patch_05": {
        "dir": ROOT / "data/raw/test_patches/patch_05_urban_dense/patch",
        "row": 7168,
        "col": 6656,
    },
}


OUTPUT_ROOT = (
    ROOT / "data/outputs/benchmark/satlas"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Create model
# ============================================================

print()
print("Loading Satlas ESRGAN...")

model = SSR_RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4,
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

state_dict = checkpoint["params_ema"]

model.load_state_dict(
    state_dict,
    strict=True,
)

model.eval()
model.to(device)

print("Satlas model loaded successfully.")


# ============================================================
# Find extracted Sentinel-2 band
# ============================================================

# ============================================================
# Find extracted Sentinel-2 band
# ============================================================

def find_band(patch_dir, band):

    # Remove the leading "B"
    # B04 -> 04
    short_name = band.replace("B", "")

    # Try all naming conventions used in our patches
    candidates = [
        patch_dir / f"{band}.tif",          # B04.tif
        patch_dir / f"{short_name}.tif",    # 04.tif
        patch_dir / f"{band}_10m.tif",      # B04_10m.tif
        patch_dir / f"{short_name}_10m.tif",# 04_10m.tif
    ]

    for path in candidates:

        if path.exists():
            return path

    # Fallback: original Sentinel-2 JP2 naming
    files = list(
        patch_dir.glob(
            f"*{band}_10m.jp2"
        )
    )

    if files:
        return files[0]

    # If nothing worked, print what actually exists
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
# Load 128 × 128 RGB patch
#
# Satlas expects:
# B04 = Red
# B03 = Green
# B02 = Blue
# ============================================================

def load_rgb_patch(
    patch_dir,
    row,
    col,
):

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

            # ------------------------------------------------
            # The extracted patch files are already 128×128.
            # For those files, read the complete image.
            #
            # For original JP2 files, use the configured
            # Sentinel-2 window.
            # ------------------------------------------------

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
# Run one patch
# ============================================================

def run_patch(
    name,
    config,
):

    print()
    print("=" * 60)
    print(f"PROCESSING {name}")
    print("=" * 60)

    rgb = load_rgb_patch(
        config["dir"],
        config["row"],
        config["col"],
    )

    print(
        "Input shape:",
        rgb.shape,
    )

    print(
        "Input range:",
        float(rgb.min()),
        "→",
        float(rgb.max()),
    )

    tensor = torch.from_numpy(
        rgb
    ).unsqueeze(0).to(
        device
    )

    print(
        "Tensor:",
        tensor.shape,
    )

    with torch.inference_mode():

        output = model(
            tensor
        )

    output = output.squeeze(
        0
    ).cpu().numpy()

    output = np.clip(
        output,
        0,
        1,
    )

    print(
        "Output shape:",
        output.shape,
    )

    print(
        "Output range:",
        float(output.min()),
        "→",
        float(output.max()),
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_ROOT / name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"satlas_{name}.tif"
    )

    # Use the B04 source for CRS / transform
    reference_path = find_band(
        config["dir"],
        "B04",
    )

    with rasterio.open(
        reference_path
    ) as src:

        profile = src.profile.copy()

        transform = src.window_transform(
            Window(
                config["col"],
                config["row"],
                128,
                128,
            )
        )

        profile.update(
            driver="GTiff",
            height=512,
            width=512,
            count=3,
            dtype="float32",
            transform=transform,
            crs=src.crs,
            compress="deflate",
        )

    # 10m → 2.5m
    transform = transform * transform.scale(
        1 / 4,
        1 / 4,
    )

    profile.update(
        transform=transform
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            output.astype(
                np.float32
            )
        )

    print()
    print("Saved:")
    print(output_path)

    # Clean CUDA memory between patches
    del tensor
    del output

    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# Run all four
# ============================================================

for name, config in PATCHES.items():

    run_patch(
        name,
        config,
    )


print()
print("=" * 60)
print("ALL SATLAS PATCHES COMPLETE")
print("=" * 60)