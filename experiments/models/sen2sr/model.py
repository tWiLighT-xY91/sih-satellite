"""
model.py
========
Public entry point for WEO-SAS/sen2sr stored on HuggingFace Hub.

All parameters are read from config.json.

Usage
-----
    from huggingface_hub import snapshot_download
    import sys

    local_dir = snapshot_download("WEO-SAS/sen2sr")
    sys.path.insert(0, local_dir)
    from model import Model

    model = Model(local_dir=local_dir)

    # Array inference: (4, H, W) float32 in [0, 1] -> (4, H*4, W*4) float32
    sr = model.predict(image)

    # GeoTIFF pipeline
    model.predict_tif("s2_scene.tif", "s2_sr.tif")
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import List, Optional

import numpy as np


def _load_module(name: str, path: str):
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Model:
    """
    Public SEN2SR model interface for HuggingFace Hub users.

    Parameters
    ----------
    local_dir : str
        Path to the directory returned by ``snapshot_download(repo_id)``.
    **overrides
        Optionally override any value from config.json, e.g.
        ``Model(local_dir=d, patch_size=256, overlap=64)``.
    """

    def __init__(self, local_dir: str, **overrides):
        config_path = os.path.join(local_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        config.update(overrides)

        if local_dir not in sys.path:
            sys.path.insert(0, local_dir)

        sen2sr_pt   = _load_module("sen2sr_pt", os.path.join(local_dir, "sen2sr_pt.py"))
        self._model = sen2sr_pt.SEN2SRPT(local_dir=local_dir, config=config)
        self.description = config.get("description", "")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run 4x super-resolution on a single image.

        Parameters
        ----------
        image : (C, H, W) float32 numpy array, values in [0, 1]
                C must equal in_channels (4 for RGBN)

        Returns
        -------
        (C, H*4, W*4) float32 numpy array
        """
        return self._model.predict(image)

    def predict_tif(
        self,
        input_path:  str,
        output_path: str,
        bands:       Optional[List[int]] = None,
    ) -> None:
        """
        Full GeoTIFF super-resolution pipeline.

        Parameters
        ----------
        input_path  : path to input Sentinel-2 GeoTIFF
        output_path : output path for the 2.5 m SR GeoTIFF
        bands       : 0-based band indices to read (default: [0, 1, 2, 3])
        """
        self._model.predict_tif(input_path, output_path, bands)
