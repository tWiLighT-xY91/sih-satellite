from pathlib import Path

from data_loader import SEN2VENUSLoader
from adapters import LDSRS2Adapter


REPO = Path(__file__).resolve().parents[2]

DATASET = (
    REPO
    / "data/raw/sen2venus/KUDALIAR.zip"
)

CHECKPOINT = (
    REPO
    / "experiments/ldsr_s2/opensr-ldsrs2_v1_0_0.ckpt"
)


print("Loading dataset...")

with SEN2VENUSLoader(DATASET) as loader:

    sample = loader.get_sample(0)

    lr = sample["lr"]

    print("LR shape:", lr.shape)
    print("LR range:", lr.min(), "→", lr.max())

    print("\nLoading LDSR-S2 adapter...")

    model = LDSRS2Adapter(
        CHECKPOINT
    )

    print("Running inference...")

    sr = model.predict(lr)

    print("\nResults:")
    print("Input :", lr.shape)
    print("Output:", sr.shape)
    print("Range :", sr.min(), "→", sr.max())

    assert lr.shape == (4, 128, 128)
    assert sr.shape == (4, 512, 512)

    print("\n✓ LDSR adapter test passed")