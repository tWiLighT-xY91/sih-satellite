"""
predictor.py
============
Inference wrapper for WEO-SAS/sen2sr (SEN2SRLite RGBN x4).

Super-resolves 4-band Sentinel-2 RGBN imagery from 10 m to 2.5 m (4x).

Usage
-----
    predictor = SEN2SRPredictor("./sen2sr")

    # Array inference: (4, H, W) float32 in [0, 1] -> (4, H*4, W*4) float32
    sr = predictor.predict(image)

    # GeoTIFF pipeline (reads Sentinel-2 DN, writes SR GeoTIFF at 2.5 m)
    predictor.predict_tif("s2_scene.tif", "s2_sr.tif", bands=[0, 1, 2, 3])

Requirements
------------
torch, numpy, rasterio, safetensors, sen2sr  (pip install sen2sr)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import rasterio


class SEN2SRPredictor:
    """
    SEN2SRLite RGBN x4 predictor.

    Parameters
    ----------
    local_dir : local path to a downloaded WEO-SAS/sen2sr model repo
    device    : torch device (auto-detected if None)
    model     : pre-built srmodel callable; bypasses weight loading (used by sen2sr_pt.py)
    """

    def __init__(
        self,
        local_dir: str,
        device:    Optional[torch.device] = None,
        model      = None,
    ):
        local_dir = Path(local_dir)
        with open(local_dir / "config.json") as f:
            cfg = json.load(f)

        self.local_dir          = local_dir
        self.in_channels        = cfg["in_channels"]
        self.out_channels       = cfg["out_channels"]
        self.scaling_factor     = cfg["scaling_factor"]
        self.patch_size         = cfg["patch_size"]
        self.overlap            = cfg["overlap"]
        self.p_low              = cfg["p_low"]
        self.p_high             = cfg["p_high"]
        self.normalization_factor = cfg["normalization_factor"]
        self.description        = cfg.get("description", "")

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if model is not None:
            self.model = model
        else:
            self._load_model(local_dir, cfg)

    # ------------------------------------------------------------------
    # Model loading (only used when model= is not injected)
    # ------------------------------------------------------------------

    def _load_model(self, local_dir: Path, cfg: dict) -> None:
        try:
            import safetensors.torch
            from sen2sr.models.opensr_baseline.cnn import CNNSR
            from sen2sr.models.tricks import HardConstraint
            from sen2sr.nonreference import srmodel
        except ImportError as exc:
            raise ImportError(
                "sen2sr and safetensors are required. "
                "Install: pip install sen2sr safetensors"
            ) from exc

        device = self.device

        weights = safetensors.torch.load_file(local_dir / cfg["weights_file"])
        sr_model = CNNSR(
            cfg["in_channels"],
            cfg["out_channels"],
            cfg["feature_channels"],
            cfg["scaling_factor"],
            cfg["bias"],
            cfg["train_mode"],
            cfg["num_blocks"],
        )
        sr_model.load_state_dict(weights)
        sr_model.to(device).eval()
        for p in sr_model.parameters():
            p.requires_grad = False

        hc_weights = safetensors.torch.load_file(local_dir / cfg["hard_constraint_file"])
        hard_constraint = HardConstraint(
            low_pass_mask=hc_weights["weights"].to(device), device=device
        )

        self.model = srmodel(sr_model, hard_constraint, device)

    # ------------------------------------------------------------------
    # Array inference
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run 4x super-resolution on a (C, H, W) float32 image.

        Uses sen2sr.predict_large for images larger than patch_size so that
        tile boundaries are blended seamlessly.

        Parameters
        ----------
        image : (C, H, W) float32, values in [0, 1]
                C must equal in_channels (4 for RGBN)

        Returns
        -------
        (C, H*4, W*4) float32 in the same radiometric range as the input
        """
        if image.ndim != 3 or image.shape[0] != self.in_channels:
            raise ValueError(
                f"Expected ({self.in_channels}, H, W), got {image.shape}"
            )

        try:
            import sen2sr
        except ImportError as exc:
            raise ImportError("pip install sen2sr") from exc

        X = torch.from_numpy(image).float().to(self.device)

        if image.shape[1] <= self.patch_size and image.shape[2] <= self.patch_size:
            with torch.no_grad():
                out = self.model(X.unsqueeze(0)).squeeze(0)   # (C, H*sf, W*sf)
        else:
            out = sen2sr.predict_large(
                model   = self.model,
                X       = X,
                overlap = self.overlap,
            )

        return out.cpu().numpy()

    # ------------------------------------------------------------------
    # GeoTIFF pipeline
    # ------------------------------------------------------------------

    def predict_tif(
        self,
        input_path:  str,
        output_path: str,
        bands:       Optional[List[int]] = None,
    ) -> None:
        """
        Full GeoTIFF super-resolution pipeline.

        Reads bands from the input GeoTIFF, normalises Sentinel-2 DN to [0, 1]
        (divides by normalization_factor if values suggest DN range, otherwise
        leaves as-is), runs 4x SR, and writes the output GeoTIFF with the
        geotransform pixel size divided by scaling_factor.

        Parameters
        ----------
        input_path  : path to input Sentinel-2 GeoTIFF
        output_path : output path for the 2.5 m SR GeoTIFF
        bands       : 0-based band indices to read (default: [0, 1, 2, 3])
        """
        bands = bands or list(range(self.in_channels))

        with rasterio.open(input_path) as src:
            arr     = src.read([b + 1 for b in bands]).astype(np.float32)
            profile = src.profile.copy()

        # Auto-normalise: if values look like raw Sentinel-2 DN (> 2.0) divide
        # by normalization_factor, otherwise assume already in [0, 1]
        if arr.max() > 2.0:
            arr = np.clip(arr / self.normalization_factor, 0.0, 1.0)

        print(
            f"SR inference  model=sen2sr  input={arr.shape}  "
            f"factor={self.scaling_factor}x  {input_path}"
        )

        sr = self.predict(arr)    # (C, H*sf, W*sf)

        print(
            f"Output shape {sr.shape}  "
            f"range [{sr.min():.4f}, {sr.max():.4f}]"
        )

        tf          = profile["transform"]
        new_tf      = tf * tf.scale(1.0 / self.scaling_factor, 1.0 / self.scaling_factor)
        out_profile = profile.copy()
        out_profile.update(
            count     = sr.shape[0],
            height    = sr.shape[1],
            width     = sr.shape[2],
            dtype     = "float32",
            transform = new_tf,
            compress  = "lzw",
        )
        out_profile.pop("photometric", None)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(sr)

        sr_res = abs(tf.a) / self.scaling_factor
        print(f"Written: {output_path}  (res={sr_res:.4f} m)")
