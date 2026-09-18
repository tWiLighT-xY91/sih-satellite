from pathlib import Path
from zipfile import ZipFile
import csv
import tempfile

import numpy as np
import rasterio


class SEN2VENUSLoader:
    """
    Lazy loader for the SEN2VENUS KUDALIAR dataset.

    The dataset is structured as:
        outer KUDALIAR.zip
            -> date/site inner ZIPs
                -> individual 10m / 5m TIFFs

    Each index.csv row represents one paired sample.
    """

    def __init__(self, archive_path):
        self.archive_path = Path(archive_path)

        if not self.archive_path.exists():
            raise FileNotFoundError(f"Dataset archive not found: {self.archive_path}")

        self.outer_zip = ZipFile(self.archive_path)

        # Read index.csv directly from the outer archive.
        with self.outer_zip.open("KUDALIAR/index.csv") as f:
            text = (line.decode("utf-8") for line in f)
            self.rows = list(
                csv.DictReader(text, delimiter="\t", skipinitialspace=True)
            )

        if not self.rows:
            raise ValueError("No samples found in index.csv")

    def __len__(self):
        return len(self.rows)

    def _read_nested_tiff(self, inner_zip_name, tiff_path):
        """
        Extract one TIFF from one inner ZIP into a temporary file,
        read it with rasterio, then remove it automatically.
        """

        outer_member = f"KUDALIAR/{inner_zip_name}"

        with self.outer_zip.open(outer_member) as inner_file:
            inner_zip_bytes = inner_file.read()

        with tempfile.TemporaryDirectory(prefix="sen2venus_") as tmpdir:

            inner_zip_path = Path(tmpdir) / "inner.zip"
            tiff_path_local = Path(tmpdir) / "sample.tif"

            inner_zip_path.write_bytes(inner_zip_bytes)

            with ZipFile(inner_zip_path) as inner_zip:
                with inner_zip.open(tiff_path) as src:
                    tiff_path_local.write_bytes(src.read())

            with rasterio.open(tiff_path_local) as src:
                data = src.read()
                profile = src.profile.copy()

        return data, profile

    @staticmethod
    def _normalize_reflectance(data):
        """
        Convert Sentinel-2 integer reflectance representation
        to float32 reflectance.

        Dataset values are stored approximately on a 0-10000 scale.
        """

        return data.astype(np.float32) / 10000.0

    def get_sample(self, index):
        """
        Load one paired LR-HR RGBN sample.

        Returns
        -------
        dict:
            {
                "index": int,
                "lr": np.ndarray,       # (4, 128, 128)
                "hr": np.ndarray,       # (4, 256, 256)
                "lr_profile": dict,
                "hr_profile": dict,
            }
        """

        if index < 0 or index >= len(self.rows):
            raise IndexError(
                f"Sample index {index} outside dataset " f"(0-{len(self.rows)-1})"
            )

        row = self.rows[index]

        lr_full_path = row["b2b3b4b8_10m"]
        hr_full_path = row["b2b3b4b8_05m"]

        # index.csv stores:
        #
        # inner.zip/path/to/file.tif
        #
        # Separate those two pieces.
        lr_zip, lr_tiff = lr_full_path.split("/", 1)
        hr_zip, hr_tiff = hr_full_path.split("/", 1)

        lr, lr_profile = self._read_nested_tiff(lr_zip, lr_tiff)

        hr, hr_profile = self._read_nested_tiff(hr_zip, hr_tiff)

        lr = self._normalize_reflectance(lr)
        hr = self._normalize_reflectance(hr)

        self._validate_pair(lr, hr, lr_profile, hr_profile)

        return {
            "index": index,
            "lr": lr,
            "hr": hr,
            "lr_profile": lr_profile,
            "hr_profile": hr_profile,
        }

    @staticmethod
    def _validate_pair(lr, hr, lr_profile, hr_profile):

        # Band count
        if lr.shape[0] != 4:
            raise ValueError(f"Expected 4 LR bands, got {lr.shape[0]}")

        if hr.shape[0] != 4:
            raise ValueError(f"Expected 4 HR bands, got {hr.shape[0]}")

        # Spatial relationship
        if hr.shape[1] != 2 * lr.shape[1]:
            raise ValueError("HR height is not 2x LR height")

        if hr.shape[2] != 2 * lr.shape[2]:
            raise ValueError("HR width is not 2x LR width")

        # CRS
        if lr_profile["crs"] != hr_profile["crs"]:
            raise ValueError("LR and HR CRS do not match")

        # Resolution
        lr_res = abs(lr_profile["transform"].a)
        hr_res = abs(hr_profile["transform"].a)

        if not np.isclose(lr_res, 2 * hr_res):
            raise ValueError(
                f"Unexpected resolution ratio: " f"LR={lr_res}, HR={hr_res}"
            )

        # Bounds
        lr_bounds = rasterio.transform.array_bounds(
            lr.shape[1], lr.shape[2], lr_profile["transform"]
        )

        hr_bounds = rasterio.transform.array_bounds(
            hr.shape[1], hr.shape[2], hr_profile["transform"]
        )

        if not np.allclose(lr_bounds, hr_bounds, atol=1e-3):
            raise ValueError("LR and HR geographic bounds do not match")

        # Values
        if not np.isfinite(lr).all():
            raise ValueError("LR contains NaN or Inf")

        if not np.isfinite(hr).all():
            raise ValueError("HR contains NaN or Inf")

    def close(self):
        self.outer_zip.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
