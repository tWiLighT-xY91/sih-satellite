"""
sen2sr_pt.py
============
HuggingFace-aware wrapper for WEO-SAS/sen2sr.

Handles all model variants by dispatching on config.json fields:
  - architecture : "cnn" | "mamba" | "swin"
  - srmodel_type : "nonreference" | "referencex2" | "referencex4"

Loads weights and builds the srmodel callable, then injects it into
SEN2SRPredictor via model= so weights are only loaded once.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import List, Optional

import numpy as np
import safetensors.torch
import torch

from base import BaseModel


def _load_module(name: str, path: str):
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _build_backbone(architecture: str, config: dict, device):
    """Instantiate and return a bare backbone (weights not loaded yet)."""
    if architecture == "cnn":
        from sen2sr.models.opensr_baseline.cnn import CNNSR
        return CNNSR(
            config["in_channels"],
            config["out_channels"],
            config["feature_channels"],
            config.get("upscale", config["scaling_factor"]),
            config["bias"],
            config["train_mode"],
            config["num_blocks"],
        )
    elif architecture == "mamba":
        from sen2sr.models.opensr_baseline.mamba import MambaSR
        return MambaSR(
            img_size          = tuple(config["img_size"]),
            in_channels       = config["in_channels"],
            out_channels      = config["out_channels"],
            embed_dim         = config["embed_dim"],
            depths            = config["depths"],
            num_heads         = config["num_heads"],
            mlp_ratio         = config["mlp_ratio"],
            upscale           = config.get("upscale", config["scaling_factor"]),
            attention_type    = config["attention_type"],
            upsampler         = config["upsampler"],
            resi_connection   = config["resi_connection"],
            operation_attention = config["operation_attention"],
        )
    elif architecture == "swin":
        from sen2sr.models.opensr_baseline.swin import Swin2SR
        return Swin2SR(
            img_size        = tuple(config["img_size"]),
            in_channels     = config["in_channels"],
            out_channels    = config["out_channels"],
            embed_dim       = config["embed_dim"],
            depths          = config["depths"],
            num_heads       = config["num_heads"],
            window_size     = config["window_size"],
            mlp_ratio       = config["mlp_ratio"],
            upscale         = config.get("upscale", 1),
            resi_connection = config["resi_connection"],
            upsampler       = config["upsampler"],
        )
    else:
        raise ValueError(f"Unknown architecture '{architecture}'")


def _freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_single_stage(local_dir: str, config: dict, device) -> object:
    """Load a single-stage srmodel (nonreference or referencex2)."""
    from sen2sr.models.tricks import HardConstraint

    arch   = config["architecture"]
    stype  = config["srmodel_type"]

    weights = safetensors.torch.load_file(
        os.path.join(local_dir, config["weights_file"])
    )
    backbone = _build_backbone(arch, config, device)
    backbone.load_state_dict(weights)
    _freeze(backbone.to(device))

    hc_weights = safetensors.torch.load_file(
        os.path.join(local_dir, config["hard_constraint_file"])
    )
    hc_kwargs = dict(
        low_pass_mask = hc_weights["weights"].to(device),
        device        = device,
    )
    if "hard_constraint_bands" in config and config["hard_constraint_bands"] is not None:
        hc_kwargs["bands"] = config["hard_constraint_bands"]
    hard_constraint = _freeze(HardConstraint(**hc_kwargs))

    if stype == "nonreference":
        from sen2sr.nonreference import srmodel
        return srmodel(backbone, hard_constraint, device)
    elif stype == "referencex2":
        from sen2sr.referencex2 import srmodel
        return srmodel(sr_model=backbone, hard_constraint=hard_constraint, device=device)
    else:
        raise ValueError(f"Unexpected srmodel_type '{stype}' for single-stage loader")


def _f2_config(config: dict) -> dict:
    """Build a per-stage config for the f2/main RSWIR backbone.
    Allows mamba-main to override embed_dim/depths/num_heads/img_size for Swin."""
    cfg = dict(config, architecture=config["f2_architecture"],
               in_channels=10, out_channels=6, upscale=1)
    for key in ("embed_dim", "depths", "num_heads", "img_size"):
        f2_key = f"f2_{key}"
        if f2_key in config:
            cfg[key] = config[f2_key]
    return cfg


def _load_referencex4(local_dir: str, config: dict, device) -> object:
    """
    Load the multi-stage referencex4 pipeline:
      Stage 1 : RGBN 10m→2.5m (sr_model.safetensor, architecture = sr_architecture)
      Stage 2 : RSWIR 20m→10m (f2_model.safetensor, architecture = f2_architecture)
      Stage 3 : RSWIR 10m→2.5m (model.safetensor,  architecture = f2_architecture)
    """
    from sen2sr.models.tricks import HardConstraint
    from sen2sr.nonreference import srmodel as rgbn_srmodel
    from sen2sr.referencex2 import srmodel as rswir_x2
    from sen2sr.referencex4 import srmodel as rswir_x4

    # -- Stage 1: RGBN backbone --
    sr_cfg = dict(config, architecture=config["sr_architecture"],
                  in_channels=4, out_channels=4, upscale=4, scaling_factor=4)
    sr_weights = safetensors.torch.load_file(os.path.join(local_dir, config["sr_weights_file"]))
    sr_backbone = _build_backbone(config["sr_architecture"], sr_cfg, device)
    sr_backbone.load_state_dict(sr_weights)
    _freeze(sr_backbone.to(device))
    sr_hc_w = safetensors.torch.load_file(os.path.join(local_dir, config["sr_hard_constraint_file"]))
    sr_hc = _freeze(HardConstraint(low_pass_mask=sr_hc_w["weights"].to(device), device=device))
    rgbn_model = rgbn_srmodel(sr_model=sr_backbone, hard_constraint=sr_hc, device=device)

    # -- Stage 2: RSWIR 20m→10m backbone --
    f2_cfg     = _f2_config(config)
    f2_weights = safetensors.torch.load_file(os.path.join(local_dir, config["f2_weights_file"]))
    f2_backbone = _build_backbone(config["f2_architecture"], f2_cfg, device)
    f2_backbone.load_state_dict(f2_weights)
    _freeze(f2_backbone.to(device))
    f2_hc_w = safetensors.torch.load_file(os.path.join(local_dir, config["f2_hard_constraint_file"]))
    f2_hc = _freeze(HardConstraint(low_pass_mask=f2_hc_w["weights"].to(device),
                                   bands=[0,1,2,3,4,5], device=device))
    rswir_model_x2 = rswir_x2(sr_model=f2_backbone, hard_constraint=f2_hc, device=device)

    # -- Stage 3: RSWIR 10m→2.5m backbone --
    main_cfg     = _f2_config(config)
    main_weights = safetensors.torch.load_file(os.path.join(local_dir, config["weights_file"]))
    main_backbone = _build_backbone(config["f2_architecture"], main_cfg, device)
    main_backbone.load_state_dict(main_weights)
    _freeze(main_backbone.to(device))
    main_hc_w = safetensors.torch.load_file(os.path.join(local_dir, config["hard_constraint_file"]))
    main_hc = _freeze(HardConstraint(low_pass_mask=main_hc_w["weights"].to(device),
                                     bands=[0,1,2,3,4,5], device=device))

    return rswir_x4(rgbn_model, rswir_model_x2, main_backbone, main_hc, device=device)


class SEN2SRPT(BaseModel):
    """
    PyTorch SEN2SR model loaded from a HuggingFace (flat) model directory.

    Parameters
    ----------
    local_dir : str — path to snapshot_download directory
    config    : dict — contents of config.json with optional user overrides
    """

    def __init__(self, local_dir: str, config: dict):
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stype    = config["srmodel_type"]

        try:
            import sen2sr  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "sen2sr and safetensors are required. "
                "Install: pip install sen2sr safetensors"
            ) from exc

        if stype in ("nonreference", "referencex2"):
            model = _load_single_stage(local_dir, config, device)
        elif stype == "referencex4":
            model = _load_referencex4(local_dir, config, device)
        else:
            raise ValueError(f"Unknown srmodel_type '{stype}'")

        predictor_mod   = _load_module("predictor", os.path.join(local_dir, "predictor.py"))
        self._predictor = predictor_mod.SEN2SRPredictor(
            local_dir = local_dir,
            device    = device,
            model     = model,
        )

        for key in ("patch_size", "overlap", "scaling_factor"):
            if key in config:
                setattr(self._predictor, key, config[key])

    def predict(self, image: np.ndarray) -> np.ndarray:
        return self._predictor.predict(image)

    def predict_tif(
        self,
        input_path:  str,
        output_path: str,
        bands:       Optional[List[int]] = None,
    ) -> None:
        self._predictor.predict_tif(input_path, output_path, bands)
