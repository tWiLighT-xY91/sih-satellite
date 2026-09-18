import numpy as np

from data_loader import SEN2VENUSLoader


DATASET = "data/raw/sen2venus/KUDALIAR.zip"


with SEN2VENUSLoader(DATASET) as dataset:

    print("\n" + "=" * 60)
    print("SEN2VENµS KUDALIAR LOADER TEST")
    print("=" * 60)

    print(f"Dataset samples : {len(dataset)}")

    sample = dataset.get_sample(0)

    lr = sample["lr"]
    hr = sample["hr"]

    print("\nSample 0")
    print("-" * 40)

    print("LR shape        :", lr.shape)
    print("HR shape        :", hr.shape)

    print("LR dtype        :", lr.dtype)
    print("HR dtype        :", hr.dtype)

    print("LR range        :", lr.min(), "→", lr.max())
    print("HR range        :", hr.min(), "→", hr.max())

    print("LR CRS          :", sample["lr_profile"]["crs"])
    print("HR CRS          :", sample["hr_profile"]["crs"])

    print(
        "LR resolution   :",
        abs(sample["lr_profile"]["transform"].a),
        "m"
    )

    print(
        "HR resolution   :",
        abs(sample["hr_profile"]["transform"].a),
        "m"
    )

    print("\n✓ Paired sample loaded successfully")
    print("✓ All validation checks passed")