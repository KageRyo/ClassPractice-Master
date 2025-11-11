import logging
from typing import Optional

import numpy as np

from src.utils.convolution import convolve_2d

logger = logging.getLogger(__name__)


def _gaussian_kernel_5x5() -> np.ndarray:
    kernel = np.array(
        [
            [1, 4, 7, 4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1, 4, 7, 4, 1],
        ],
        dtype=np.float64,
    )
    return kernel / 273.0


class UnsharpMaskingOperator:
    """Unsharp masking with manually implemented smoothing and scaling."""

    def __init__(self, amount: float = 1.0, smoothing_kernel: Optional[np.ndarray] = None):
        if amount <= 0:
            raise ValueError('Unsharp masking amount must be positive')
        self.amount = float(amount)
        self.kernel = smoothing_kernel.copy() if smoothing_kernel is not None else _gaussian_kernel_5x5()
        logger.info('UnsharpMaskingOperator initialized (amount=%.2f)', self.amount)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError('Unsharp masking expects a 2D grayscale image')
        image_float = np.asarray(image, dtype=np.float64)
        blurred = convolve_2d(image_float, self.kernel)
        mask = image_float - blurred
        sharpened = image_float + self.amount * mask
        sharpened = np.clip(np.rint(sharpened), 0, 255).astype(np.uint8)
        logger.debug('Unsharp masking completed (amount=%.2f)', self.amount)
        return sharpened


def apply_unsharp_masking(image: np.ndarray, amount: float = 1.0, smoothing_kernel: Optional[np.ndarray] = None) -> np.ndarray:
    """Functional helper for unsharp masking."""
    operator = UnsharpMaskingOperator(amount=amount, smoothing_kernel=smoothing_kernel)
    return operator.apply(image)
