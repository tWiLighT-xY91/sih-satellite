import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _validate_inputs(pred, target):
    """
    Validate prediction and target arrays.

    Expected shape:
        (C, H, W)

    Both arrays must have identical dimensions.
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction {pred.shape}, "
            f"target {target.shape}"
        )

    if pred.ndim != 3:
        raise ValueError(
            f"Expected (C, H, W), got {pred.shape}"
        )

    return pred, target


def psnr(pred, target, data_range=1.0):
    """
    Peak Signal-to-Noise Ratio.

    Higher is better.
    """
    pred, target = _validate_inputs(pred, target)

    return peak_signal_noise_ratio(
        target,
        pred,
        data_range=data_range
    )


def ssim(pred, target, data_range=1.0):
    """
    Structural Similarity Index.

    Computes SSIM independently for each band
    and returns the mean.

    Higher is better.
    """
    pred, target = _validate_inputs(pred, target)

    scores = []

    for band in range(pred.shape[0]):
        score = structural_similarity(
            target[band],
            pred[band],
            data_range=data_range
        )
        scores.append(score)

    return float(np.mean(scores))


def sam(pred, target, eps=1e-8):
    """
    Spectral Angle Mapper.

    Measures the angle between predicted and
    target spectral vectors.

    Lower is better.

    Returns mean angle in degrees.
    """
    pred, target = _validate_inputs(pred, target)

    # (C, H, W) -> (H, W, C)
    pred = np.moveaxis(pred, 0, -1)
    target = np.moveaxis(target, 0, -1)

    dot = np.sum(pred * target, axis=-1)

    pred_norm = np.linalg.norm(pred, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)

    denominator = pred_norm * target_norm + eps

    cosine = dot / denominator

    # Numerical safety
    cosine = np.clip(cosine, -1.0, 1.0)

    angles = np.arccos(cosine)

    # Convert radians → degrees
    angles = np.degrees(angles)

    # Ignore pixels where both spectra are effectively zero
    valid = (pred_norm > eps) & (target_norm > eps)

    if not np.any(valid):
        return 0.0

    return float(np.mean(angles[valid]))


def ergas(pred, target, ratio=1.0, eps=1e-8):
    """
    ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse).

    Lower is better.

    ratio:
        Resolution ratio between reference and
        reconstructed image.

        Here both prediction and target are on the
        5m grid, so ratio = 1.
    """
    pred, target = _validate_inputs(pred, target)

    band_scores = []

    for band in range(pred.shape[0]):

        target_mean = np.mean(target[band])

        if abs(target_mean) < eps:
            continue

        rmse = np.sqrt(
            np.mean((pred[band] - target[band]) ** 2)
        )

        relative_error = rmse / (target_mean + eps)

        band_scores.append(relative_error ** 2)

    if not band_scores:
        return 0.0

    return float(
        100.0
        * (ratio ** 0.5)
        * np.sqrt(np.mean(band_scores))
    )


def calculate_all_metrics(pred, target, data_range=1.0):
    """
    Calculate the complete evaluation metric set.

    Returns
    -------
    dict
        Dictionary containing PSNR, SSIM, SAM and ERGAS.
    """

    pred, target = _validate_inputs(pred, target)

    return {
        "PSNR": psnr(pred, target, data_range),
        "SSIM": ssim(pred, target, data_range),
        "SAM": sam(pred, target),
        "ERGAS": ergas(pred, target),
    }