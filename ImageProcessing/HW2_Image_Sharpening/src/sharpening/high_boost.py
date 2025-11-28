import logging
from typing import Optional

import numpy as np

from src.utils.convolution import convolve_2d
from src.sharpening.unsharp_mask import _gaussian_kernel_5x5

logger = logging.getLogger(__name__)


class HighBoostFilteringOperator:
    """High-boost filtering using manual smoothing and scaling."""

    def __init__(self, boost_factor: float = 1.5, smoothing_kernel: Optional[np.ndarray] = None):
        if boost_factor <= 1.0:
            raise ValueError('High-boost factor must be greater than 1.0')
        self.boost_factor = float(boost_factor)
        if smoothing_kernel is not None:
            self.kernel = smoothing_kernel.copy()
        else:
            self.kernel = _gaussian_kernel_5x5()
        logger.info('HighBoostFilteringOperator initialized (factor=%.2f)', self.boost_factor)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError('High-boost filtering expects a 2D grayscale image')
        image_float = np.asarray(image, dtype=np.float64)
        blurred = convolve_2d(image_float, self.kernel)
        mask = image_float - blurred
        sharpened = image_float + (self.boost_factor - 1.0) * mask
        sharpened = np.clip(np.rint(sharpened), 0, 255).astype(np.uint8)
        logger.debug('High-boost filtering completed (factor=%.2f)', self.boost_factor)
        return sharpened


def apply_high_boost_filter(image: np.ndarray, boost_factor: float = 1.5, smoothing_kernel: Optional[np.ndarray] = None) -> np.ndarray:
    """Functional helper for high-boost filtering."""
    operator = HighBoostFilteringOperator(boost_factor=boost_factor, smoothing_kernel=smoothing_kernel)
    return operator.apply(image)
