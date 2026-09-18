import numpy as np


def aggregate_2x_to_5m(sr_2_5m):
    """
    Aggregate a 2.5m SR prediction to the 5m grid.

    Expected input:
        (C, H, W)

    Output:
        (C, H/2, W/2)

    A 2x2 mean pooling operation is used.
    """

    sr_2_5m = np.asarray(sr_2_5m, dtype=np.float32)

    if sr_2_5m.ndim != 3:
        raise ValueError(
            f"Expected (C, H, W), got {sr_2_5m.shape}"
        )

    channels, height, width = sr_2_5m.shape

    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            f"Spatial dimensions must be even, got "
            f"{height}x{width}"
        )

    output = sr_2_5m.reshape(
        channels,
        height // 2,
        2,
        width // 2,
        2,
    ).mean(axis=(2, 4))

    return output


def validate_prediction_target(prediction, target):
    """
    Ensure the prediction and reference are compatible
    for metric calculation.
    """

    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/reference shape mismatch: "
            f"{prediction.shape} vs {target.shape}"
        )

    if prediction.ndim != 3:
        raise ValueError(
            f"Expected (C, H, W), got {prediction.shape}"
        )

    return True