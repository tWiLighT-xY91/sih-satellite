from pathlib import Path

from features import (
    extract_scene_features,
    print_features,
)


ROOT = (
    Path.home()
    / "sih"
    / "sih-satellite"
)

RAW_ROOT = (
    ROOT
    / "data/raw/test_patches"
)


PATCHES = {

    "patch_01": (
        RAW_ROOT
        / "patch_01_forest/patch"
    ),

    "patch_02": (
        RAW_ROOT
        / "patch_02_urban/patch"
    ),

    "patch_03": (
        RAW_ROOT
        / "patch_03_agriculture/patch"
    ),

    "patch_04": (
        RAW_ROOT
        / "patch_04_atmospheric/patch"
    ),

    "patch_05": (
        RAW_ROOT
        / "patch_05_urban_dense/patch"
    ),
}


for patch_id, patch_dir in PATCHES.items():

    print()
    print("#" * 70)
    print(patch_id)
    print("#" * 70)

    features = extract_scene_features(
        patch_dir
    )

    print_features(
        features
    )