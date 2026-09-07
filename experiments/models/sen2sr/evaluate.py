"""
sen2sr_evaluate.py
==================
Evaluate WEO-SAS/sen2sr variants using the opensr-test benchmark suite.

Metrics computed per variant × dataset:
  - reflectance  (↓)  L1 distance — radiometric fidelity
  - spectral     (↓)  Spectral Angle Distance — colour consistency
  - spatial      (↓)  Phase Correlation — geometric stability
  - synthesis    (↑)  High-frequency detail added
  - hallucination(↓)  False details not in HR
  - omission     (↓)  Real details missing from SR
  - improvement  (↑)  Correct new details introduced

Usage
-----
    pip install opensr-test huggingface_hub sen2sr safetensors rasterio

    # Evaluate everything (RGBN-compatible datasets + variants)
    python sen2sr_evaluate.py

    # Specific variants and/or datasets
    python sen2sr_evaluate.py --variants main mamba-rgbn-x4 --datasets naip spot

    # Skip download if already cached
    python sen2sr_evaluate.py --cache-dir ./model_cache

Notes
-----
- RGBN variants (main, lite-rgbn-x4, mamba-rgbn-x4) are evaluated on all
  opensr-test datasets (NAIP, SPOT, Venus, Spain Crops, Spain Urban).
- Full-pipeline 10-band variants (lite-main, mamba-main) and RSWIR variants
  (lite-rswir-x2, mamba-rswir-x2) require all 10 Sentinel-2 bands.
  opensr-test only provides 4-band RGBN patches, so these variants use the
  4 RGBN bands for input and the remaining 6 channels are zero-padded.
  For a fair evaluation of those variants, use your own 10-band Sentinel-2
  tiles and call evaluate_custom() directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANTS: Dict[str, dict] = {
    "main": {
        "repo_id":     "WEO-SAS/sen2sr",
        "revision":    None,
        "in_channels": 4,
        "scale":       4,
        "note":        "SEN2SRLite RGBN 4x (CNN)",
    },
    "lite-rswir-x2": {
        "repo_id":     "WEO-SAS/sen2sr",
        "revision":    "lite-rswir-x2",
        "in_channels": 10,
        "scale":       2,
        "note":        "SEN2SRLite RSWIR 2x (CNN) — zero-pads channels 4-9",
    },
    "lite-main": {
        "repo_id":     "WEO-SAS/sen2sr",
        "revision":    "lite-main",
        "in_channels": 10,
        "scale":       4,
        "note":        "SEN2SRLite full 10-band 4x (CNN) — zero-pads channels 4-9",
    },
    "mamba-rgbn-x4": {
        "repo_id":     "WEO-SAS/sen2sr",
        "revision":    "mamba-rgbn-x4",
        "in_channels": 4,
        "scale":       4,
        "note":        "SEN2SR RGBN 4x (Mamba)",
    },
    "mamba-rswir-x2": {
        "repo_id":     "WEO-SAS/sen2sr",
        "revision":    "mamba-rswir-x2",
        "in_channels": 10,
        "scale":       2,
        "note":        "SEN2SR RSWIR 2x (Swin2SR) — zero-pads channels 4-9",
    },
    "mamba-main": {
        "repo_id":     "WEO-SAS/sen2sr",
        "revision":    "mamba-main",
        "in_channels": 10,
        "scale":       4,
        "note":        "SEN2SR full 10-band 4x (Mamba+Swin) — zero-pads channels 4-9",
    },
    "srresnet": {
        "repo_id":     "WEO-SAS/srresnet",
        "revision":    None,
        "in_channels": 4,
        "scale":       4,
        "note":        "SRResNet RGBN→RGB 4x (baseline)",
    },
}

DATASETS = ["naip", "spot", "venus", "spain_crops", "spain_urban"]

# Canonical output column names → actual opensr_test.Metrics key
METRIC_MAP = {
    "reflectance":  "reflectance",
    "spectral":     "spectral",
    "spatial":      "spatial",
    "synthesis":    "synthesis",
    "hallucination": "ha_metric",
    "omission":     "om_metric",
    "improvement":  "im_metric",
}
METRIC_COLS = list(METRIC_MAP.keys())


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(variant: str, cache_dir: str, local_models_dir: Optional[str] = None):
    """Load a WEO-SAS model variant from a local dir or by downloading from HF Hub."""
    if local_models_dir:
        local_dir = str(Path(local_models_dir) / variant)
        if not Path(local_dir).is_dir():
            raise FileNotFoundError(f"Model dir not found: {local_dir}")
    else:
        from huggingface_hub import snapshot_download
        repo_id  = VARIANTS[variant].get("repo_id", "WEO-SAS/sen2sr")
        revision = VARIANTS[variant]["revision"]
        kwargs   = dict(repo_id=repo_id, local_dir=f"{cache_dir}/{variant}")
        if revision:
            kwargs["revision"] = revision
        local_dir = snapshot_download(**kwargs)

    sys.path.insert(0, local_dir)

    # Clear any cached module from a previous variant
    for mod in ["model", "sen2sr_pt", "predictor", "base"]:
        sys.modules.pop(mod, None)

    from model import Model  # noqa: PLC0415
    return Model(local_dir=local_dir)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _pad_to_multiple(arr: np.ndarray, multiple: int) -> tuple:
    """Pad (C, H, W) to the next multiple of `multiple`; return (padded, orig_h, orig_w)."""
    _, h, w = arr.shape
    h_pad = ((h + multiple - 1) // multiple) * multiple
    w_pad = ((w + multiple - 1) // multiple) * multiple
    if h_pad == h and w_pad == w:
        return arr, h, w
    padded = np.zeros((arr.shape[0], h_pad, w_pad), dtype=arr.dtype)
    padded[:, :h, :w] = arr
    return padded, h, w


def run_sr(model, lr_np: np.ndarray, in_channels: int, scale: int = 4,
           patch_size: int = 128) -> np.ndarray:
    """
    Run SR on a single LR patch.

    lr_np   : (C_avail, H, W) float32 in [0, 1]   — opensr-test provides C=4 (RGBN)
    Returns : (C_out, H*scale, W*scale) float32, cropped to exact expected size
    """
    C_avail = lr_np.shape[0]

    if in_channels == C_avail:
        inp = lr_np
    elif in_channels > C_avail:
        pad = np.zeros((in_channels - C_avail,) + lr_np.shape[1:], dtype=np.float32)
        inp = np.concatenate([lr_np, pad], axis=0)
    else:
        inp = lr_np[:in_channels]

    # Pre-pad to patch_size so HardConstraint sees consistent LR↔SR sizes
    orig_h, orig_w = inp.shape[1], inp.shape[2]
    inp, _, _ = _pad_to_multiple(inp, patch_size)

    sr = model.predict(inp)

    # Crop to exact expected size based on original (unpadded) LR dimensions
    h_out = orig_h * scale
    w_out = orig_w * scale
    return sr[:, :h_out, :w_out]


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def _save_comparison(
    lr: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    path: Path,
    title: str,
    variant: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage.transform import resize as sk_resize

        def to_rgb(arr):
            rgb = np.clip(arr[:3].transpose(1, 2, 0), 0, 1)
            return (rgb * 255).astype(np.uint8)

        hr_h, hr_w = hr.shape[1], hr.shape[2]
        lr_big = sk_resize(to_rgb(lr), (hr_h, hr_w), order=1, preserve_range=True).astype(np.uint8)
        sr_rgb = to_rgb(sr)
        hr_rgb = to_rgb(hr)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, img, label in zip(
            axes,
            [lr_big, sr_rgb, hr_rgb],
            ["LR (bicubic)", f"SR ({variant})", "HR (reference)"],
        ):
            ax.imshow(img)
            ax.set_title(label, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{variant} — {title}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved image: {path.name}")
    except Exception as e:
        print(f"    [WARN] Could not save image: {e}")


def evaluate_dataset(
    model,
    variant:     str,
    dataset_name: str,
    max_samples: Optional[int] = None,
    save_images_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Run a variant against one opensr-test dataset and return mean metrics.

    Returns a dict mapping metric name → mean value, or empty dict on error.
    """
    try:
        import opensr_test
    except ImportError:
        raise ImportError("pip install opensr-test")

    try:
        dataset = opensr_test.load(dataset_name)
    except Exception as e:
        print(f"    [WARN] Could not load dataset '{dataset_name}': {e}")
        return {}

    # opensr-test dataset is a dict: {"L2A": (N,C,H,W) uint16, "HRharm": (N,C,H,W) uint16}
    lr_all = dataset["L2A"]
    hr_all = dataset["HRharm"]

    metrics_obj = opensr_test.Metrics()
    vinfo       = VARIANTS[variant]
    in_ch       = vinfo["in_channels"]
    scale       = vinfo["scale"]
    accum: Dict[str, list] = {m: [] for m in METRIC_COLS}
    n = lr_all.shape[0] if max_samples is None else min(max_samples, lr_all.shape[0])
    saved_image = False

    for i in range(n):
        lr = lr_all[i].astype(np.float32) / 10000.0  # (C, H, W) → [0, 1]
        hr = hr_all[i].astype(np.float32) / 10000.0

        try:
            sr = run_sr(model, lr, in_ch, scale)
        except Exception as e:
            print(f"    [WARN] SR failed on sample {i}: {e}")
            continue

        # For x2 models on 4x datasets: SR is half the HR size — skip metrics
        if sr.shape[1] != hr.shape[1] or sr.shape[2] != hr.shape[2]:
            if i == 0:
                print(f"    [SKIP] SR {sr.shape} != HR {hr.shape} — scale mismatch, skipping dataset")
            continue

        if save_images_dir and not saved_image:
            img_path = save_images_dir / f"{variant}_{dataset_name}.png"
            _save_comparison(lr, sr, hr, img_path, dataset_name, variant)
            saved_image = True

        lr_t = torch.from_numpy(lr)
        sr_t = torch.from_numpy(sr)
        hr_t = torch.from_numpy(hr)

        # Align channels: metrics require lr/sr/hr to have the same count
        min_ch = min(lr_t.shape[0], sr_t.shape[0], hr_t.shape[0])
        lr_t, sr_t, hr_t = lr_t[:min_ch], sr_t[:min_ch], hr_t[:min_ch]

        try:
            result = metrics_obj.compute(lr=lr_t, sr=sr_t, hr=hr_t)
            if not isinstance(result, dict):
                result = vars(result) if hasattr(result, "__dict__") else {}
        except Exception as e:
            print(f"    [WARN] Metrics failed on sample {i}: {e}")
            continue

        for col, api_key in METRIC_MAP.items():
            val = result.get(api_key)
            if val is not None:
                v = float(val.mean()) if hasattr(val, "mean") else float(val)
                accum[col].append(v)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n} samples processed", end="\r")

    print()
    return {m: float(np.mean(vs)) if vs else float("nan") for m, vs in accum.items()}


# ---------------------------------------------------------------------------
# HF output helpers
# ---------------------------------------------------------------------------

def _nan_to_null(obj):
    """Recursively replace float NaN with None so json.dump produces valid JSON."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_null(v) for v in obj]
    return obj


def build_eval_json(rows: list) -> dict:
    """Build eval_results.json dict from accumulated CSV rows."""
    from collections import defaultdict

    per_dataset: dict = {}
    agg: dict = defaultdict(lambda: defaultdict(list))

    for row in rows:
        v   = row["variant"]
        ds  = row["dataset"]
        per_dataset.setdefault(ds, {})
        m_vals = {}
        for m in METRIC_COLS:
            val = row.get(m, float("nan"))
            m_vals[m] = val
            if not (isinstance(val, float) and np.isnan(val)):
                agg[v][m].append(val)
        per_dataset[ds][v] = m_vals

    aggregate = {
        v: {m: float(np.mean(vs)) if vs else float("nan") for m, vs in metrics.items()}
        for v, metrics in agg.items()
    }

    variants_meta = {
        v: {"note": VARIANTS[v]["note"], "in_channels": VARIANTS[v]["in_channels"],
            "scale": VARIANTS[v]["scale"]}
        for v in VARIANTS
        if v in agg
    }

    return {
        "eval_type": "super_resolution",
        "model_name": "SEN2SR",
        "variants": variants_meta,
        "per_dataset": per_dataset,
        "aggregate": aggregate,
    }


def push_to_hf(
    eval_json: dict,
    images_dir: Optional[Path],
    csv_path: str,
    hf_token: str,
    commit_message: str = "eval: update benchmark results",
) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)

    repo_id = "WEO-SAS/sen2sr"

    # Push eval_results.json
    eval_str = json.dumps(_nan_to_null(eval_json), indent=2)
    api.upload_file(
        path_or_fileobj=eval_str.encode(),
        path_in_repo="eval_results.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print("Pushed eval_results.json")

    # Push CSV
    if Path(csv_path).exists():
        api.upload_file(
            path_or_fileobj=csv_path,
            path_in_repo=f"eval/{Path(csv_path).name}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        print(f"Pushed eval/{Path(csv_path).name}")

    # Push images
    if images_dir and images_dir.exists():
        for img_path in sorted(images_dir.glob("*.png")):
            api.upload_file(
                path_or_fileobj=str(img_path),
                path_in_repo=f"eval_images/{img_path.name}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
            print(f"Pushed eval_images/{img_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate WEO-SAS/sen2sr variants")
    parser.add_argument(
        "--variants", nargs="+", default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
        help="Variants to evaluate (default: all)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS, choices=DATASETS,
        help="Datasets to use (default: all)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap samples per dataset (useful for a quick smoke-test)",
    )
    parser.add_argument(
        "--cache-dir", default="./sen2sr_model_cache",
        help="Directory to cache downloaded model weights",
    )
    parser.add_argument(
        "--local-models-dir", default=None,
        help="Use pre-downloaded models instead of HF Hub (subdir per variant: main/, lite-main/, etc.)",
    )
    parser.add_argument(
        "--output", default="sen2sr_eval_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--images-dir", default="./eval_images",
        help="Directory for visual comparison PNG files",
    )
    parser.add_argument(
        "--hf-token", default=None,
        help="HuggingFace write token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--no-push", action="store_true",
        help="Skip HF push (dry-run)",
    )
    args = parser.parse_args()

    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    rows = []

    for variant in args.variants:
        print(f"\n{'='*60}")
        print(f"Variant: {variant}  ({VARIANTS[variant]['note']})")
        print(f"{'='*60}")

        try:
            print("  Loading model...")
            model = load_model(variant, args.cache_dir, args.local_models_dir)
        except Exception as e:
            print(f"  [ERROR] Could not load model: {e}")
            continue

        for ds in args.datasets:
            print(f"  Dataset: {ds}")
            metrics = evaluate_dataset(model, variant, ds, args.max_samples, images_dir)
            if not metrics:
                continue

            row = {"variant": variant, "dataset": ds}
            row.update(metrics)
            rows.append(row)

            # Pretty-print
            print(f"  {'Metric':<16} {'Value':>10}")
            print(f"  {'-'*28}")
            for m in METRIC_COLS:
                arrow = "↑" if m in ("synthesis", "improvement") else "↓"
                print(f"  {m:<16} {metrics.get(m, float('nan')):>9.4f} {arrow}")

    # Save CSV + eval_results.json
    if rows:
        fieldnames = ["variant", "dataset"] + METRIC_COLS
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to: {args.output}")

        eval_json = build_eval_json(rows)
        json_path = Path(args.output).parent / "eval_results.json"
        with open(json_path, "w") as f:
            json.dump(_nan_to_null(eval_json), f, indent=2)
        print(f"Results saved to: {json_path}")
    else:
        print("\nNo results to save.")

    # Summary table
    if rows:
        print("\n" + "="*60)
        print("SUMMARY — mean across all datasets")
        print("="*60)
        from collections import defaultdict
        agg: dict = defaultdict(lambda: defaultdict(list))
        for row in rows:
            for m in METRIC_COLS:
                v = row.get(m, float("nan"))
                if not np.isnan(v):
                    agg[row["variant"]][m].append(v)

        header = f"{'Variant':<20}" + "".join(f"{m[:8]:>11}" for m in METRIC_COLS)
        print(header)
        print("-" * len(header))
        for variant in args.variants:
            if variant not in agg:
                continue
            vals = "".join(
                f"{np.mean(agg[variant].get(m, [float('nan')])):>11.4f}"
                for m in METRIC_COLS
            )
            print(f"{variant:<20}{vals}")

    # Push to HF
    if rows and not args.no_push:
        if not hf_token:
            print("\n[WARN] No HF token — skipping push. Pass --hf-token or set HF_TOKEN.")
        else:
            print("\nPushing results to HuggingFace...")
            push_to_hf(eval_json, images_dir, args.output, hf_token)
            print("Done.")


if __name__ == "__main__":
    main()
