import logging
import numpy as np

logger = logging.getLogger(__name__)


class WienerFilterOperator:
    """Wiener filter (MMSE) for image restoration."""

    def __init__(
        self,
        k: float = 0.0025,
        noise_variance: float = 100.0,
        signal_variance: float = None,
    ):
        if k < 0:
            raise ValueError('System parameter k must be non-negative')
        if noise_variance < 0:
            raise ValueError('Noise variance must be non-negative')
        
        self.k = float(k)
        self.noise_variance = float(noise_variance)
        self.signal_variance = signal_variance
        
        logger.info(
            'WienerFilterOperator initialized (k=%.6f, noise_variance=%.2f)',
            self.k, self.noise_variance
        )

    def _compute_degradation_function(self, rows: int, cols: int) -> np.ndarray:
        """Compute H(u,v) = exp(-k * (u^2 + v^2)^(5/6))."""
        u_coords = np.arange(cols) - cols / 2.0
        v_coords = np.arange(rows) - rows / 2.0
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)
        
        distance_squared = u_grid ** 2 + v_grid ** 2
        epsilon = 1e-10
        distance_squared = np.maximum(distance_squared, epsilon)
        
        exponent = -self.k * np.power(distance_squared, 5.0 / 6.0)
        degradation_function = np.exp(exponent)
        
        return degradation_function

    def _estimate_nsr(self, image: np.ndarray) -> float:
        """Estimate noise-to-signal ratio (NSR = noise_var / signal_var)."""
        if self.signal_variance is not None:
            signal_var = self.signal_variance
        else:
            signal_var = np.var(image)
            if signal_var < 1e-6:
                signal_var = 1.0
        
        nsr = self.noise_variance / signal_var
        logger.debug('Estimated NSR: %.6f (noise_var=%.2f, signal_var=%.2f)',
                     nsr, self.noise_variance, signal_var)
        return nsr

    def apply(self, degraded_image: np.ndarray) -> np.ndarray:
        """Apply Wiener filter: F_hat = [H* / (|H|^2 + K)] * G."""
        if degraded_image.ndim != 2:
            raise ValueError('Wiener filtering expects a 2D grayscale image')
        
        rows, cols = degraded_image.shape
        image_float = np.asarray(degraded_image, dtype=np.float64)
        
        K = self._estimate_nsr(image_float)
        
        frequency_domain = np.fft.fft2(image_float)
        frequency_shifted = np.fft.fftshift(frequency_domain)
        
        H = self._compute_degradation_function(rows, cols)
        H_conj = np.conj(H)
        H_magnitude_squared = np.abs(H) ** 2
        
        wiener_filter = H_conj / (H_magnitude_squared + K)
        restored_frequency = frequency_shifted * wiener_filter
        
        restored_shifted = np.fft.ifftshift(restored_frequency)
        restored_spatial = np.fft.ifft2(restored_shifted)
        restored_real = np.real(restored_spatial)
        
        # Percentile normalization to handle outliers
        p_low, p_high = np.percentile(restored_real, (1, 99))
        
        if p_high > p_low:
            normalized = (restored_real - p_low) / (p_high - p_low) * 255.0
            normalized = np.clip(normalized, 0, 255)
        else:
            logger.warning('Wiener filter produced near-constant output')
            normalized = np.full_like(restored_real, np.mean(degraded_image))
        
        output_uint8 = np.clip(np.rint(normalized), 0, 255).astype(np.uint8)
        logger.debug('Wiener filtering completed')
        
        return output_uint8


def apply_wiener_filter(
    degraded_image: np.ndarray,
    k: float = 0.0025,
    noise_variance: float = 100.0,
    signal_variance: float = None,
) -> np.ndarray:
    """Convenience function for Wiener filtering."""
    operator = WienerFilterOperator(
        k=k,
        noise_variance=noise_variance,
        signal_variance=signal_variance,
    )
    return operator.apply(degraded_image)
