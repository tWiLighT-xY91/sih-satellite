import numpy as np

from metrics import calculate_all_metrics


np.random.seed(42)

# Synthetic 4-band reference image
target = np.random.rand(4, 64, 64).astype(np.float32)

# Slightly perturbed prediction
noise = np.random.normal(
    0,
    0.01,
    target.shape
).astype(np.float32)

prediction = target + noise

metrics = calculate_all_metrics(
    prediction,
    target
)

print("\n" + "=" * 50)
print("METRIC SANITY CHECK")
print("=" * 50)

for name, value in metrics.items():
    print(f"{name:10s}: {value:.6f}")