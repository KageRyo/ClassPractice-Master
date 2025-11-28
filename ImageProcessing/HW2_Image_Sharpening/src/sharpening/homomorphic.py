import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class HomomorphicFilteringOperator:
    """Homomorphic filter implemented in the frequency domain with NumPy FFT."""

    def __init__(
        self,
        gamma_l: float = 0.8,
        gamma_h: float = 1.5,
        cutoff_frequency: float = 30.0,
        c: float = 1.0,
    ):
        if gamma_l <= 0 or gamma_h <= 0:
            raise ValueError('Gamma values must be positive')
        if gamma_h <= gamma_l:
            raise ValueError(
                'gamma_h must be greater than gamma_l for high-pass emphasis')
        if cutoff_frequency <= 0:
            raise ValueError('Cutoff frequency must be positive')
        if c <= 0:
            raise ValueError('Shape parameter c must be positive')
        self.gamma_l = float(gamma_l)
        self.gamma_h = float(gamma_h)
        self.cutoff_frequency = float(cutoff_frequency)
        self.c = float(c)
        logger.info(
            'HomomorphicFilteringOperator initialized (gamma_l=%.2f, gamma_h=%.2f, cutoff=%.2f, c=%.2f)',
            self.gamma_l,
            self.gamma_h,
            self.cutoff_frequency,
            self.c,
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError(
                'Homomorphic filtering expects a 2D grayscale image')
        image_float = np.asarray(image, dtype=np.float64)
        normalized = image_float / 255.0
        normalized = np.clip(normalized, 1e-6, None)

        log_image = np.log(normalized)
        frequency = np.fft.fft2(log_image)
        frequency_shifted = np.fft.fftshift(frequency)

        rows, cols = image.shape
        row_indices = np.arange(rows) - rows / 2.0
        col_indices = np.arange(cols) - cols / 2.0
        v, u = np.meshgrid(row_indices, col_indices, indexing='ij')
        distance_sq = u ** 2 + v ** 2
        cutoff_sq = self.cutoff_frequency ** 2

        smoothing_factor = np.exp(-(self.c * distance_sq) / cutoff_sq)
        homomorphic_filter = (self.gamma_h - self.gamma_l) * \
            (1.0 - smoothing_factor) + self.gamma_l

        filtered_frequency = homomorphic_filter * frequency_shifted
        spatial_shifted = np.fft.ifftshift(filtered_frequency)
        spatial_log = np.fft.ifft2(spatial_shifted)
        spatial_real = np.real(spatial_log)

        exp_result = np.exp(spatial_real)
        exp_result = np.clip(exp_result, 1e-6, None)

        min_val = float(exp_result.min())
        max_val = float(exp_result.max())
        if not math.isfinite(min_val) or not math.isfinite(max_val) or max_val <= min_val:
            logger.warning(
                'Homomorphic filter produced degenerate range; returning zeros image')
            return np.zeros_like(image, dtype=np.uint8)

        normalized_output = (exp_result - min_val) / (max_val - min_val)
        output_uint8 = np.clip(
            np.rint(normalized_output * 255.0), 0, 255).astype(np.uint8)
        logger.debug('Homomorphic filtering completed')
        return output_uint8


def apply_homomorphic_filter(
    image: np.ndarray,
    gamma_l: float = 0.8,
    gamma_h: float = 1.5,
    cutoff_frequency: float = 30.0,
    c: float = 1.0,
) -> np.ndarray:
    """Functional helper for homomorphic filtering."""
    operator = HomomorphicFilteringOperator(
        gamma_l=gamma_l,
        gamma_h=gamma_h,
        cutoff_frequency=cutoff_frequency,
        c=c,
    )
    return operator.apply(image)
