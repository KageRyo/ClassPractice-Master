import logging
import numpy as np

logger = logging.getLogger(__name__)


class InverseFilterOperator:
    """Inverse filter with low-pass constraint for image restoration."""

    def __init__(
        self,
        k: float = 0.0025,
        cutoff_radius: float = 50.0,
        epsilon: float = 1e-6,
    ):
        if k < 0:
            raise ValueError('System parameter k must be non-negative')
        if cutoff_radius <= 0:
            raise ValueError('Cutoff radius must be positive')
        if epsilon <= 0:
            raise ValueError('Epsilon must be positive')
        
        self.k = float(k)
        self.cutoff_radius = float(cutoff_radius)
        self.epsilon = float(epsilon)
        
        logger.info(
            'InverseFilterOperator initialized (k=%.6f, cutoff_radius=%.2f, epsilon=%.2e)',
            self.k, self.cutoff_radius, self.epsilon
        )

    def _compute_degradation_function(self, rows: int, cols: int) -> np.ndarray:
        """Compute H(u,v) = exp(-k * (u^2 + v^2)^(5/6))."""
        u_coords = np.arange(cols) - cols / 2.0
        v_coords = np.arange(rows) - rows / 2.0
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)
        
        distance_squared = u_grid ** 2 + v_grid ** 2
        distance_squared = np.maximum(distance_squared, self.epsilon)
        
        exponent = -self.k * np.power(distance_squared, 5.0 / 6.0)
        degradation_function = np.exp(exponent)
        
        return degradation_function

    def _compute_lowpass_filter(self, rows: int, cols: int) -> np.ndarray:
        """Compute Gaussian low-pass filter."""
        u_coords = np.arange(cols) - cols / 2.0
        v_coords = np.arange(rows) - rows / 2.0
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)
        
        distance = np.sqrt(u_grid ** 2 + v_grid ** 2)
        lowpass_filter = np.exp(-(distance ** 2) / (2 * self.cutoff_radius ** 2))
        
        return lowpass_filter

    def apply(self, degraded_image: np.ndarray) -> np.ndarray:
        if degraded_image.ndim != 2:
            raise ValueError('Inverse filtering expects a 2D grayscale image')
        
        rows, cols = degraded_image.shape
        image_float = np.asarray(degraded_image, dtype=np.float64)
        
        frequency_domain = np.fft.fft2(image_float)
        frequency_shifted = np.fft.fftshift(frequency_domain)
        
        H = self._compute_degradation_function(rows, cols)
        lowpass = self._compute_lowpass_filter(rows, cols)
        
        # Regularized inverse: H / (|H|^2 + alpha)
        alpha = 0.01
        H_magnitude_squared = np.abs(H) ** 2
        regularized_inverse = H / (H_magnitude_squared + alpha)
        
        combined_filter = regularized_inverse * lowpass + (1.0 - lowpass)
        restored_frequency = frequency_shifted * combined_filter
        
        restored_shifted = np.fft.ifftshift(restored_frequency)
        restored_spatial = np.fft.ifft2(restored_shifted)
        restored_real = np.real(restored_spatial)
        
        # Percentile normalization to handle outliers
        p_low, p_high = np.percentile(restored_real, (1, 99))
        
        if p_high > p_low:
            normalized = (restored_real - p_low) / (p_high - p_low) * 255.0
            normalized = np.clip(normalized, 0, 255)
        else:
            logger.warning('Inverse filter produced near-constant output')
            normalized = np.full_like(restored_real, np.mean(degraded_image))
        
        output_uint8 = np.clip(np.rint(normalized), 0, 255).astype(np.uint8)
        logger.debug('Inverse filtering completed')
        
        return output_uint8


def apply_inverse_filter(
    degraded_image: np.ndarray,
    k: float = 0.0025,
    cutoff_radius: float = 50.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Convenience function for inverse filtering."""
    operator = InverseFilterOperator(
        k=k,
        cutoff_radius=cutoff_radius,
        epsilon=epsilon,
    )
    return operator.apply(degraded_image)
