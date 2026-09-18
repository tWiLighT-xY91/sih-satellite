from pathlib import Path

from data_loader import SEN2VENUSLoader
from adapters import MambaSEN2SRAdapter


REPO = Path(__file__).resolve().parents[2]

DATASET = (
    REPO
    / "data/raw/sen2venus/KUDALIAR.zip"
)

MODEL_DIR = (
    REPO
    / "checkpoints/sen2sr-mamba-rgbn-x4"
)


print("Loading dataset...")

with SEN2VENUSLoader(DATASET) as loader:

    sample = loader.get_sample(0)

    lr = sample["lr"]

    print("LR shape:", lr.shape)
    print("LR range:", lr.min(), "→", lr.max())

    print("\nLoading Mamba-SEN2SR adapter...")

    model = MambaSEN2SRAdapter(
        MODEL_DIR
    )

    print("Running inference...")

    sr = model.predict(lr)

    print("\nResults:")
    print("Input :", lr.shape)
    print("Output:", sr.shape)
    print("Range :", sr.min(), "→", sr.max())

    assert lr.shape == (4, 128, 128)
    assert sr.shape == (4, 512, 512)

    print("\n✓ Mamba-SEN2SR adapter test passed")