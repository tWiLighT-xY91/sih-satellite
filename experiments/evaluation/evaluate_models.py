from pathlib import Path
import csv
import time

import numpy as np

from data_loader import SEN2VENUSLoader
from adapters import (
    LDSRS2Adapter,
    MambaSEN2SRAdapter,
    SEN2SRCNNAdapter,
)
from alignment import aggregate_2x_to_5m
from metrics import calculate_all_metrics


REPO = Path(__file__).resolve().parents[2]

DATASET = (
    REPO
    / "data/raw/sen2venus/KUDALIAR.zip"
)

LDSR_CHECKPOINT = (
    REPO
    / "experiments/ldsr_s2/opensr-ldsrs2_v1_0_0.ckpt"
)

MAMBA_DIR = (
    REPO
    / "checkpoints/sen2sr-mamba-rgbn-x4"
)

CNN_DIR = (
    REPO
    / "checkpoints/sen2sr-cnn-rgbn-x4"
)

OUTPUT_DIR = (
    REPO
    / "data/outputs/evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_PATH = OUTPUT_DIR / "kudaliar_smoke_test.csv"

NUM_SAMPLES = 10


def main():

    print("=" * 70)
    print("SEN2VENµS MULTI-MODEL SMOKE BENCHMARK")
    print("=" * 70)

    print("\nLoading models...")

    ldsr = LDSRS2Adapter(
        LDSR_CHECKPOINT
    )

    cnn = SEN2SRCNNAdapter(
        CNN_DIR
    )

    mamba = MambaSEN2SRAdapter(
        MAMBA_DIR
    )

    models = {
        "LDSR-S2": ldsr,
        "SEN2SR-CNN": cnn,
        "Mamba-SEN2SR": mamba,
    }

    print("\n✓ All three models loaded.")

    rows = []

    with SEN2VENUSLoader(DATASET) as loader:

        total = min(
            NUM_SAMPLES,
            len(loader),
        )

        print(
            f"\nEvaluating {total} samples..."
        )

        for index in range(total):

            print(
                f"\n[{index + 1}/{total}] "
                f"Sample {index}"
            )

            sample = loader.get_sample(index)

            lr = sample["lr"]
            hr = sample["hr"]

            print(
                f"  LR: {lr.shape} | "
                f"HR: {hr.shape}"
            )

            for model_name, model in models.items():

                print(
                    f"  → {model_name}"
                )

                start = time.perf_counter()

                sr_25m = model.predict(lr)

                inference_time = (
                    time.perf_counter()
                    - start
                )

                # 2.5m → 5m
                sr_5m = aggregate_2x_to_5m(
                    sr_25m
                )

                # Final shape validation
                if sr_5m.shape != hr.shape:
                    raise RuntimeError(
                        f"{model_name}: "
                        f"prediction shape "
                        f"{sr_5m.shape} != "
                        f"HR shape {hr.shape}"
                    )

                metrics = calculate_all_metrics(
                    sr_5m,
                    hr,
                )

                row = {
                    "sample": index,
                    "model": model_name,
                    "PSNR": metrics["PSNR"],
                    "SSIM": metrics["SSIM"],
                    "SAM": metrics["SAM"],
                    "ERGAS": metrics["ERGAS"],
                    "inference_seconds": inference_time,
                }

                rows.append(row)

                print(
                    f"     PSNR={metrics['PSNR']:.4f}  "
                    f"SSIM={metrics['SSIM']:.4f}  "
                    f"SAM={metrics['SAM']:.4f}  "
                    f"ERGAS={metrics['ERGAS']:.4f}  "
                    f"time={inference_time:.2f}s"
                )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    fieldnames = [
        "sample",
        "model",
        "PSNR",
        "SSIM",
        "SAM",
        "ERGAS",
        "inference_seconds",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print("\nResults saved to:")
    print(CSV_PATH)

    print(
        f"\nTotal evaluations: {len(rows)}"
    )


if __name__ == "__main__":
    main()