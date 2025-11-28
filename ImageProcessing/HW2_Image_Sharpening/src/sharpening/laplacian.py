import logging
from typing import Literal

import numpy as np

from src.utils.convolution import convolve_2d

logger = logging.getLogger(__name__)


class LaplacianSharpeningOperator:
    """Apply Laplacian sharpening with manually defined kernels."""

    def __init__(self, kernel_type: Literal['8-connected', '4-connected'] = '8-connected', alpha: float = 1.0):
        self.kernel_type = kernel_type
        self.alpha = float(alpha)
        self.kernel = self._select_kernel(kernel_type)
        logger.info('LaplacianSharpeningOperator initialized (%s kernel, alpha=%.2f)', kernel_type, self.alpha)

    @staticmethod
    def _select_kernel(kernel_type: str) -> np.ndarray:
        if kernel_type == '8-connected':
            return np.array(
                [[-1, -1, -1],
                 [-1,  8, -1],
                 [-1, -1, -1]],
                dtype=np.float64,
            )
        if kernel_type == '4-connected':
            return np.array(
                [[0, -1, 0],
                 [-1, 4, -1],
                 [0, -1, 0]],
                dtype=np.float64,
            )
        raise ValueError(f'Unknown Laplacian kernel type: {kernel_type}')

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError('Laplacian sharpening expects a 2D grayscale image')
        image_float = np.asarray(image, dtype=np.float64)
        laplacian_response = convolve_2d(image_float, self.kernel)
        sharpened = image_float + self.alpha * laplacian_response
        sharpened = np.clip(np.rint(sharpened), 0, 255).astype(np.uint8)
        logger.debug('Laplacian sharpening completed with alpha=%.2f', self.alpha)
        return sharpened


def apply_laplacian_sharpening(image: np.ndarray, kernel_type: Literal['8-connected', '4-connected'] = '8-connected', alpha: float = 1.0) -> np.ndarray:
    """Functional wrapper for LaplacianSharpeningOperator."""
    operator = LaplacianSharpeningOperator(kernel_type=kernel_type, alpha=alpha)
    return operator.apply(image)
