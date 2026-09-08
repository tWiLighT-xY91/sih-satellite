from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BaseModel(ABC):

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run super-resolution inference on a single image.

        Args:
            image: (C, H, W) float32 numpy array, values in [0, 1]
                   C must match in_channels (4 for RGBN variant)

        Returns:
            (C, H*scaling_factor, W*scaling_factor) float32 numpy array
            in the same radiometric range as the input
        """
        pass

    @abstractmethod
    def predict_tif(
        self,
        input_path:  str,
        output_path: str,
        bands:       Optional[List[int]] = None,
    ) -> None:
        """
        Full GeoTIFF super-resolution pipeline.

        Args:
            input_path  : path to input GeoTIFF
            output_path : output path for super-resolved GeoTIFF
            bands       : 0-based band indices to read (default: [0,1,2,3])
        """
        pass
