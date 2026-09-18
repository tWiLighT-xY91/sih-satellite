from pathlib import Path
from typing import Dict, Optional

from features import extract_scene_features


# Models currently in our SR system
MODELS = [
    "LDSR-S2",
    "Mamba-SEN2SR",
    "SEN2SR-CNN",
]


def classify_scene(features: Dict[str, float]) -> str:
    """
    Temporary scene classifier.

    This is NOT the final learned classifier.
    It is only used to test the pipeline until
    benchmark-derived training data is available.
    """

    vegetation = features["vegetation_fraction"]
    edges = features["edge_density"]
    ndvi = features["ndvi_mean"]
    spatial_var = features["spatial_variance"]

    # Strong vegetation -> agricultural / vegetated scene
    if vegetation > 0.80 and ndvi > 0.35:
        return "agriculture"

    # Very low spatial variation -> atmospheric / low-information scene
    if spatial_var < 0.00005 and edges < 0.001:
        return "low_information"

    # Strong edge density -> urban / infrastructure-heavy scene
    if edges > 0.015:
        return "urban"

    # Moderate vegetation and texture -> mixed / peri-urban
    if vegetation > 0.40 and edges > 0.005:
        return "mixed"

    return "natural"


def route_scene(
    patch_dir: str | Path,
    model_performance: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    Route a scene to an SR model.

    Parameters
    ----------
    patch_dir:
        Directory containing Sentinel-2 RGBN bands.

    model_performance:
        Future benchmark-derived performance table.

        Example:
        {
            "agriculture": {
                "LDSR-S2": 28.4,
                "Mamba-SEN2SR": 29.1,
                "SEN2SR-CNN": 27.8,
            }
        }

        Higher score = better performance.

        This will eventually be generated from our
        paired LR-HR benchmark.

    Returns
    -------
    dict
        Scene features, scene class, and selected model.
    """

    patch_dir = Path(patch_dir)

    features = extract_scene_features(patch_dir)

    scene_class = classify_scene(features)

    # ---------------------------------------------------------
    # FUTURE LEARNED ROUTER
    # ---------------------------------------------------------
    # Once benchmark results exist, the learned classifier
    # will replace this section.
    #
    # For now, use benchmark performance if supplied.
    # ---------------------------------------------------------

    selected_model = None

    if model_performance is not None:
        scene_scores = model_performance.get(scene_class, {})

        valid_scores = {
            model: score
            for model, score in scene_scores.items()
            if model in MODELS
        }

        if valid_scores:
            selected_model = max(
                valid_scores,
                key=valid_scores.get
            )

    # Temporary fallback
    if selected_model is None:
        selected_model = "Mamba-SEN2SR"

    return {
        "scene_class": scene_class,
        "selected_model": selected_model,
        "features": features,
    }