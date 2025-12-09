"""
Inverse Filtering for Image Restoration

實現直接逆濾波（結合低通濾波器）來還原退化影像。

退化模型：G(u,v) = H(u,v) * F(u,v) + N(u,v)
退化函數：H(u,v) = exp(-k * (u^2 + v^2)^(5/6))

逆濾波估計：F̂(u,v) = G(u,v) / H(u,v)

為避免 H(u,v) 趨近於 0 時的數值不穩定，結合低通濾波器限制逆濾波的頻率範圍。
"""

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
        """
        初始化逆濾波運算元。
        
        Args:
            k: 退化函數的系統參數
            cutoff_radius: 低通濾波器的截止頻率半徑（限制逆濾波範圍）
            epsilon: 防止除零的小常數
        """
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
        distance_squared = np.maximum(distance_squared, self.epsilon)
        
        # 計算 H(u,v) = exp(-k * (u^2 + v^2)^(5/6))
        exponent = -self.k * np.power(distance_squared, 5.0 / 6.0)
        degradation_function = np.exp(exponent)
        
        return degradation_function

    def _compute_lowpass_filter(self, rows: int, cols: int) -> np.ndarray:
        """
        計算理想低通濾波器，用於限制逆濾波的頻率範圍。
        
        使用高斯低通濾波器以獲得平滑的頻率響應。
        """
        u_coords = np.arange(cols) - cols / 2.0
        v_coords = np.arange(rows) - rows / 2.0
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)
        
        distance = np.sqrt(u_grid ** 2 + v_grid ** 2)
        
        # 高斯低通濾波器
        lowpass_filter = np.exp(-(distance ** 2) / (2 * self.cutoff_radius ** 2))
        
        return lowpass_filter

    def apply(self, degraded_image: np.ndarray) -> np.ndarray:
        """
        對退化影像應用逆濾波復原。
        
        Args:
            degraded_image: 退化的灰階影像（2D numpy array）
            
        Returns:
            復原後的影像（uint8 格式）
        """
        if degraded_image.ndim != 2:
            raise ValueError('Inverse filtering expects a 2D grayscale image')
        
        rows, cols = degraded_image.shape
        image_float = np.asarray(degraded_image, dtype=np.float64)
        
        # 執行 2D FFT 並中心化
        frequency_domain = np.fft.fft2(image_float)
        frequency_shifted = np.fft.fftshift(frequency_domain)
        
        # 計算退化函數
        H = self._compute_degradation_function(rows, cols)
        
        # 計算低通濾波器
        lowpass = self._compute_lowpass_filter(rows, cols)
        
        # 逆濾波結合低通濾波器：
        # 在低通範圍內進行逆濾波，高頻部分直接保留原始頻譜
        # F̂(u,v) = G(u,v) / H(u,v) * L(u,v) + G(u,v) * (1 - L(u,v))
        # 簡化為：F̂(u,v) = G(u,v) * [L(u,v)/H(u,v) + (1-L(u,v))]
        
        # 使用更穩健的逆濾波方式：避免在 H 很小時產生極端值
        # 採用正則化逆濾波：1/H -> H / (|H|^2 + α)
        alpha = 0.01  # 正則化參數
        H_magnitude_squared = np.abs(H) ** 2
        regularized_inverse = H / (H_magnitude_squared + alpha)
        
        # 應用低通限制的正則化逆濾波
        combined_filter = regularized_inverse * lowpass + (1.0 - lowpass)
        
        # 應用濾波器
        restored_frequency = frequency_shifted * combined_filter
        
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
            # 使用原始影像的均值作為輸出
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
    """
    逆濾波的便捷函數。
    
    Args:
        degraded_image: 退化的灰階影像
        k: 退化函數的系統參數
        cutoff_radius: 低通濾波器的截止頻率半徑
        epsilon: 防止除零的小常數
        
    Returns:
        復原後的影像
    """
    operator = InverseFilterOperator(
        k=k,
        cutoff_radius=cutoff_radius,
        epsilon=epsilon,
    )
    return operator.apply(degraded_image)
