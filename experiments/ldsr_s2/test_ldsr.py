import torch
import opensr_model
from omegaconf import OmegaConf
from pathlib import Path


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# --------------------------------------------------
# 2. Find installed LDSR-S2 configuration
# --------------------------------------------------

package_dir = Path(opensr_model.__file__).parent
config_path = package_dir / "configs" / "config_10m.yaml"

print(f"Config: {config_path}")

config = OmegaConf.load(config_path)

print("Configuration loaded.")


# --------------------------------------------------
# 3. Create LDSR-S2 model
# --------------------------------------------------

model = opensr_model.SRLatentDiffusion(
    config,
    device=device
)

print("Model created.")


# --------------------------------------------------
# 4. Load pretrained weights
# --------------------------------------------------

print("Loading pretrained checkpoint...")

checkpoint_path = Path("opensr-ldsrs2_v1_0_0.ckpt")

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path.resolve()}"
    )

print(f"Checkpoint: {checkpoint_path.resolve()}")

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

# The checkpoint contains the model state dictionary
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=False)

model.eval()

print("Pretrained LDSR-S2 weights loaded.")

assert model.training is False


# --------------------------------------------------
# 5. Create test input
# --------------------------------------------------

lr = torch.rand(
    1,
    4,
    128,
    128,
    device=device
)

print(f"Input shape: {lr.shape}")


# --------------------------------------------------
# 6. Run super-resolution
# --------------------------------------------------

print("Running LDSR-S2...")

with torch.no_grad():
    sr = model.forward(
        lr,
        sampling_steps=100
    )

print(f"Output shape: {sr.shape}")

print("\n🔥 LDSR-S2 INFERENCE SUCCESSFUL 🔥")