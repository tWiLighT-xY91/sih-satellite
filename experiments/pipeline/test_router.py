from pathlib import Path

from router import route_scene


PATCHES = {
    "patch_02": "data/raw/test_patches/patch_02_urban/patch",
    "patch_03": "data/raw/test_patches/patch_03_agriculture/patch",
    "patch_04": "data/raw/test_patches/patch_04_atmospheric/patch",
    "patch_05": "data/raw/test_patches/patch_05_urban_dense/patch",
}


for name, patch_path in PATCHES.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    result = route_scene(Path(patch_path))

    print(f"Scene class    : {result['scene_class']}")
    print(f"Selected model : {result['selected_model']}")