from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from opensr_model import SRLatentDiffusion


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]

if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
    
SEN2SR_ROOT = EXPERIMENTS_ROOT / "models" / "sen2sr"

if str(SEN2SR_ROOT) not in sys.path:
    sys.path.insert(0, str(SEN2SR_ROOT))


class LDSRS2Adapter:
    """
    LDSR-S2 adapter for the common evaluation pipeline.

    Input:
        numpy array, shape (4, H, W)
        band order: B02, B03, B04, B08
        values: normalized reflectance [approximately 0, 1]

    Output:
        numpy array, shape (4, 4H, 4W)
        band order: B02, B03, B04, B08
    """

    def __init__(self, checkpoint_path, device=None, sampling_steps=100):

        self.checkpoint_path = Path(checkpoint_path)
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.sampling_steps = sampling_steps

        # Load official LDSR-S2 configuration
        import opensr_model

        package_dir = Path(opensr_model.__file__).parent

        config_path = package_dir / "configs" / "config_10m.yaml"

        config = OmegaConf.load(config_path)

        # Create model
        self.model = SRLatentDiffusion(config, device=self.device)

        # Load pretrained checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        weights = checkpoint["state_dict"]

        # Match official loader behaviour
        for key in list(weights.keys()):
            if "loss" in key:
                del weights[key]

        self.model.model.load_state_dict(weights, strict=True)

        self.model.eval()

    def predict(self, lr):
        """
        Run LDSR-S2 inference.

        Parameters
        ----------
        lr : np.ndarray
            Shape (4, H, W), normalized reflectance.

        Returns
        -------
        np.ndarray
            Shape (4, 4H, 4W).
        """

        if not isinstance(lr, np.ndarray):
            raise TypeError("lr must be a numpy array")

        if lr.ndim != 3:
            raise ValueError(f"Expected lr shape (C,H,W), got {lr.shape}")

        if lr.shape[0] != 4:
            raise ValueError(f"Expected 4 bands, got {lr.shape[0]}")

        if not np.isfinite(lr).all():
            raise ValueError("Input contains NaN or infinite values")

        # IMPORTANT:
        # Data loader already performs /10000 normalization.
        tensor = torch.from_numpy(lr.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            sr = self.model.forward(tensor, sampling_steps=self.sampling_steps)

        sr = sr.squeeze(0).detach().cpu().numpy().astype(np.float32)

        expected_shape = (4, lr.shape[1] * 4, lr.shape[2] * 4)

        if sr.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected LDSR output shape: "
                f"{sr.shape}, expected {expected_shape}"
            )

        return sr


import json


class MambaSEN2SRAdapter:
    """
    Mamba-SEN2SR RGBN x4 adapter.

    Common benchmark order:
        B02, B03, B04, B08

    SEN2SR/Mamba model order:
        B04, B03, B02, B08

    Input:
        (4, H, W), normalized reflectance

    Output:
        (4, 4H, 4W), returned in canonical
        B02, B03, B04, B08 order.
    """

    def __init__(self, local_dir, device=None):

        self.local_dir = Path(local_dir)

        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = torch.device(device)

        config_path = self.local_dir / "config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        # Sanity checks: this adapter is specifically for RGBN x4 Mamba.
        if config.get("architecture") != "mamba":
            raise ValueError(
                f"Expected Mamba architecture, got " f"{config.get('architecture')}"
            )

        if config.get("in_channels") != 4:
            raise ValueError(
                f"Expected 4 input channels, got " f"{config.get('in_channels')}"
            )

        if config.get("out_channels") != 4:
            raise ValueError(
                f"Expected 4 output channels, got " f"{config.get('out_channels')}"
            )

        if config.get("scaling_factor") != 4:
            raise ValueError(
                f"Expected 4x scaling, got " f"{config.get('scaling_factor')}"
            )

        # Existing project wrapper handles:
        # MambaSR construction
        # safetensor loading
        # HardConstraint
        # SEN2SRPredictor
        from models.sen2sr.sen2sr_pt import SEN2SRPT

        self.predictor = SEN2SRPT(
            local_dir=str(self.local_dir),
            config=config,
        )

    def predict(self, lr):
        """
        Run Mamba-SEN2SR inference.

        Parameters
        ----------
        lr : np.ndarray
            (4, H, W), canonical order:
            B02, B03, B04, B08

        Returns
        -------
        np.ndarray
            (4, 4H, 4W), canonical order:
            B02, B03, B04, B08
        """

        if not isinstance(lr, np.ndarray):
            raise TypeError("lr must be a numpy array")

        if lr.ndim != 3:
            raise ValueError(f"Expected (C,H,W), got {lr.shape}")

        if lr.shape[0] != 4:
            raise ValueError(f"Expected 4 bands, got {lr.shape[0]}")

        if not np.isfinite(lr).all():
            raise ValueError("Input contains NaN or infinite values")

        # Canonical:
        # [B02, B03, B04, B08]
        #
        # Mamba/SEN2SR:
        # [B04, B03, B02, B08]
        model_input = lr[[2, 1, 0, 3]]

        sr = self.predictor.predict(model_input.astype(np.float32))

        sr = np.asarray(sr, dtype=np.float32)

        expected_shape = (
            4,
            lr.shape[1] * 4,
            lr.shape[2] * 4,
        )

        if sr.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected Mamba output shape: "
                f"{sr.shape}, expected {expected_shape}"
            )

        # Restore canonical order:
        # [B04,B03,B02,B08] → [B02,B03,B04,B08]
        sr = sr[[2, 1, 0, 3]]

        return sr

class SEN2SRCNNAdapter:
    """
    SEN2SR CNN RGBN x4 adapter.

    Common benchmark order:
        B02, B03, B04, B08

    SEN2SR model order:
        B04, B03, B02, B08

    Input:
        (4, H, W), normalized reflectance

    Output:
        (4, 4H, 4W), returned in canonical
        B02, B03, B04, B08 order.
    """

    def __init__(self, local_dir, device=None):

        self.local_dir = Path(local_dir)

        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = torch.device(device)

        config_path = self.local_dir / "config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        # Sanity checks: this adapter is specifically for CNN RGBN x4.
        if config.get("architecture") != "cnn":
            raise ValueError(
                f"Expected CNN architecture, got "
                f"{config.get('architecture')}"
            )

        if config.get("in_channels") != 4:
            raise ValueError(
                f"Expected 4 input channels, got "
                f"{config.get('in_channels')}"
            )

        if config.get("out_channels") != 4:
            raise ValueError(
                f"Expected 4 output channels, got "
                f"{config.get('out_channels')}"
            )

        if config.get("scaling_factor") != 4:
            raise ValueError(
                f"Expected 4x scaling, got "
                f"{config.get('scaling_factor')}"
            )

        from models.sen2sr.sen2sr_pt import SEN2SRPT

        self.predictor = SEN2SRPT(
            local_dir=str(self.local_dir),
            config=config,
        )

    def predict(self, lr):
        """
        Run SEN2SR CNN inference.

        Parameters
        ----------
        lr : np.ndarray
            (4, H, W), canonical order:
            B02, B03, B04, B08

        Returns
        -------
        np.ndarray
            (4, 4H, 4W), canonical order:
            B02, B03, B04, B08
        """

        if not isinstance(lr, np.ndarray):
            raise TypeError("lr must be a numpy array")

        if lr.ndim != 3:
            raise ValueError(
                f"Expected (C,H,W), got {lr.shape}"
            )

        if lr.shape[0] != 4:
            raise ValueError(
                f"Expected 4 bands, got {lr.shape[0]}"
            )

        if not np.isfinite(lr).all():
            raise ValueError(
                "Input contains NaN or infinite values"
            )

        # Canonical:
        # [B02, B03, B04, B08]
        #
        # SEN2SR:
        # [B04, B03, B02, B08]
        model_input = lr[[2, 1, 0, 3]]

        sr = self.predictor.predict(
            model_input.astype(np.float32)
        )

        sr = np.asarray(
            sr,
            dtype=np.float32
        )

        expected_shape = (
            4,
            lr.shape[1] * 4,
            lr.shape[2] * 4,
        )

        if sr.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected CNN output shape: "
                f"{sr.shape}, expected {expected_shape}"
            )

        # Restore canonical order:
        # [B04,B03,B02,B08] → [B02,B03,B04,B08]
        sr = sr[[2, 1, 0, 3]]

        return sr