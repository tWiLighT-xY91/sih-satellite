import numpy as np

from alignment import (
    aggregate_2x_to_5m,
    validate_prediction_target,
)


# Simulate a 2.5m 4-band SR output
sr = np.random.rand(
    4, 256, 256
).astype(np.float32)

# Simulate the 5m reference
hr = np.random.rand(
    4, 128, 128
).astype(np.float32)


print("\n" + "=" * 60)
print("ALIGNMENT TEST")
print("=" * 60)

print("SR input shape :", sr.shape)

sr_5m = aggregate_2x_to_5m(sr)

print("SR 5m shape    :", sr_5m.shape)
print("HR shape       :", hr.shape)

validate_prediction_target(
    sr_5m,
    hr
)

print("\n✓ 2.5m → 5m aggregation successful")
print("✓ Prediction/reference shapes match")