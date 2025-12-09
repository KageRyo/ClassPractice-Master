"""
Wiener Filtering for Image Restoration

實現最小均方誤差（維納）濾波來還原退化影像。

退化模型：G(u,v) = H(u,v) * F(u,v) + N(u,v)
退化函數：H(u,v) = exp(-k * (u^2 + v^2)^(5/6))

維納濾波估計：
F̂(u,v) = [H*(u,v) / (|H(u,v)|^2 + K)] * G(u,v)

其中：
- H*(u,v) 是 H(u,v) 的共軛複數
- K = Sn/Sf 是雜訊與信號功率譜的比值（可簡化為常數）
"""

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
        """
        初始化維納濾波運算元。
        
        Args:
            k: 退化函數的系統參數
            noise_variance: 雜訊方差（對於標準差為10的高斯雜訊，方差為100）
            signal_variance: 信號方差估計（如果為 None，則使用影像自身估計）
        """
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
        """
        計算退化函數 H(u,v) = exp(-k * (u^2 + v^2)^(5/6))
        
        使用中心化的頻率座標。
        """
        # 建立中心化的頻率座標網格
        u_coords = np.arange(cols) - cols / 2.0
        v_coords = np.arange(rows) - rows / 2.0
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)
        
        # 計算 (u^2 + v^2)^(5/6)
        distance_squared = u_grid ** 2 + v_grid ** 2
        # 避免在原點出現問題
        epsilon = 1e-10
        distance_squared = np.maximum(distance_squared, epsilon)
        
        # 計算 H(u,v) = exp(-k * (u^2 + v^2)^(5/6))
        exponent = -self.k * np.power(distance_squared, 5.0 / 6.0)
        degradation_function = np.exp(exponent)
        
        return degradation_function

    def _estimate_nsr(self, image: np.ndarray) -> float:
        """
        估計雜訊與信號功率譜比值 (NSR = Sn/Sf)。
        
        使用簡化的估計方法：K = noise_variance / signal_variance
        """
        if self.signal_variance is not None:
            signal_var = self.signal_variance
        else:
            # 使用影像方差作為信號方差的估計
            signal_var = np.var(image)
            if signal_var < 1e-6:
                signal_var = 1.0  # 防止除零
        
        nsr = self.noise_variance / signal_var
        logger.debug('Estimated NSR: %.6f (noise_var=%.2f, signal_var=%.2f)',
                     nsr, self.noise_variance, signal_var)
        return nsr

    def apply(self, degraded_image: np.ndarray) -> np.ndarray:
        """
        對退化影像應用維納濾波復原。
        
        維納濾波公式：
        F̂(u,v) = [H*(u,v) / (|H(u,v)|^2 + K)] * G(u,v)
        
        Args:
            degraded_image: 退化的灰階影像（2D numpy array）
            
        Returns:
            復原後的影像（uint8 格式）
        """
        if degraded_image.ndim != 2:
            raise ValueError('Wiener filtering expects a 2D grayscale image')
        
        rows, cols = degraded_image.shape
        image_float = np.asarray(degraded_image, dtype=np.float64)
        
        # 估計 NSR
        K = self._estimate_nsr(image_float)
        
        # 執行 2D FFT 並中心化
        frequency_domain = np.fft.fft2(image_float)
        frequency_shifted = np.fft.fftshift(frequency_domain)
        
        # 計算退化函數 H(u,v)
        H = self._compute_degradation_function(rows, cols)
        
        # 計算 H 的共軛複數（由於 H 是實數，H* = H）
        H_conj = np.conj(H)
        
        # 計算 |H|^2
        H_magnitude_squared = np.abs(H) ** 2
        
        # 維納濾波：F̂ = [H* / (|H|^2 + K)] * G
        # 這等價於：F̂ = [1/H * |H|^2 / (|H|^2 + K)] * G
        wiener_filter = H_conj / (H_magnitude_squared + K)
        
        # 應用維納濾波
        restored_frequency = frequency_shifted * wiener_filter
        
        # 逆 FFT
        restored_shifted = np.fft.ifftshift(restored_frequency)
        restored_spatial = np.fft.ifft2(restored_shifted)
        restored_real = np.real(restored_spatial)
        
        # 使用百分位數正規化，避免極端值影響
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
    """
    維納濾波的便捷函數。
    
    Args:
        degraded_image: 退化的灰階影像
        k: 退化函數的系統參數
        noise_variance: 雜訊方差（標準差為10時，方差為100）
        signal_variance: 信號方差估計
        
    Returns:
        復原後的影像
    """
    operator = WienerFilterOperator(
        k=k,
        noise_variance=noise_variance,
        signal_variance=signal_variance,
    )
    return operator.apply(degraded_image)
