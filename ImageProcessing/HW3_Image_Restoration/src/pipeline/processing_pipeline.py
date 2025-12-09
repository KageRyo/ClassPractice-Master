import logging
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.schemas.restoration_results_schema import RestorationResultsSchema
from src.restoration.inverse_filter import apply_inverse_filter
from src.restoration.wiener_filter import apply_wiener_filter
from src.ui.visualization import ImageRestorationVisualizer
from src.utils.image_utils import ImageFileLoader


@dataclass
class RestorationParameters:
    """Parameters for image restoration algorithms."""
    # 退化函數參數
    k: float = 0.0025  # 系統參數
    
    # 逆濾波參數
    inverse_cutoff_radius: float = 50.0  # 低通濾波器截止頻率
    inverse_epsilon: float = 1e-6  # 防止除零的小常數
    
    # 維納濾波參數
    noise_variance: float = 100.0  # 雜訊方差（標準差為10時，方差為100）
    signal_variance: float = None  # 信號方差（None 時自動估計）

    def summarize(self) -> str:
        return (
            f"Degradation: H(u,v)=exp(-k*(u²+v²)^(5/6)), k={self.k:.6f} | "
            f"Inverse: cutoff={self.inverse_cutoff_radius:.1f}, ε={self.inverse_epsilon:.2e} | "
            f"Wiener: noise_var={self.noise_variance:.1f}"
        )


def sanitize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float input to uint8 safely."""
    return np.clip(np.rint(np.asarray(image, dtype=np.float64)), 0, 255).astype(np.uint8)


def compute_restoration_outputs(
    degraded_image: np.ndarray,
    params: RestorationParameters,
    logger: logging.Logger,
) -> RestorationResultsSchema:
    """
    對退化影像執行兩種復原方法。
    
    Args:
        degraded_image: 退化的灰階影像
        params: 復原參數
        logger: 日誌記錄器
        
    Returns:
        包含兩種復原結果的 schema
    """
    image_uint8 = sanitize_to_uint8(degraded_image)

    logger.info('  Applying inverse filtering (k=%.6f, cutoff=%.1f)...',
                params.k, params.inverse_cutoff_radius)
    inverse_image = apply_inverse_filter(
        image_uint8,
        k=params.k,
        cutoff_radius=params.inverse_cutoff_radius,
        epsilon=params.inverse_epsilon,
    )

    logger.info('  Applying Wiener filtering (k=%.6f, noise_var=%.1f)...',
                params.k, params.noise_variance)
    wiener_image = apply_wiener_filter(
        image_uint8,
        k=params.k,
        noise_variance=params.noise_variance,
        signal_variance=params.signal_variance,
    )

    results = RestorationResultsSchema(
        inverse_filtered=inverse_image,
        wiener_filtered=wiener_image,
    )
    return results


def visualize_results(
    image_filename: str,
    original_image: np.ndarray,
    degraded_image: np.ndarray,
    results: RestorationResultsSchema,
    visualizer: ImageRestorationVisualizer,
    figure_dir: str = 'results',
    display_plot_immediately: bool = True,
) -> str:
    """
    產生比較圖並儲存。
    
    Args:
        image_filename: 原始影像檔名
        original_image: 原始影像
        degraded_image: 退化影像
        results: 復原結果
        visualizer: 視覺化工具
        figure_dir: 圖形儲存目錄
        display_plot_immediately: 是否立即顯示
        
    Returns:
        比較圖路徑
    """
    base_name = os.path.splitext(image_filename)[0]
    os.makedirs(figure_dir, exist_ok=True)
    comparison_figure_path = os.path.join(
        figure_dir, f'{base_name}_restoration_comparison.png')
    visualizer.display_restoration_results(
        image_filename=image_filename,
        original_image=sanitize_to_uint8(original_image),
        degraded_image=sanitize_to_uint8(degraded_image),
        results=results,
        figure_save_path=comparison_figure_path,
        display_plot_immediately=display_plot_immediately,
    )
    return comparison_figure_path


def save_results(
    image_filename: str,
    results: RestorationResultsSchema,
    loader: ImageFileLoader,
) -> Dict[str, str]:
    """儲存復原結果影像。"""
    base_name = os.path.splitext(image_filename)[0]
    saved_paths = {
        'inverse_filtered': loader.save_image_array_to_file(
            results.inverse_filtered, f'{base_name}_inverse.bmp'),
        'wiener_filtered': loader.save_image_array_to_file(
            results.wiener_filtered, f'{base_name}_wiener.bmp'),
    }
    return saved_paths


def collect_intensity_statistics(
    results: RestorationResultsSchema,
    original_image: np.ndarray,
    degraded_image: np.ndarray,
) -> Dict[str, float]:
    """收集影像強度統計資訊。"""
    def mean_intensity(image: np.ndarray) -> float:
        return float(np.mean(image))

    stats = {
        'original_mean': mean_intensity(sanitize_to_uint8(original_image)),
        'degraded_mean': mean_intensity(sanitize_to_uint8(degraded_image)),
        'inverse_mean': mean_intensity(results.inverse_filtered),
        'wiener_mean': mean_intensity(results.wiener_filtered),
    }
    return stats


def compute_psnr(original: np.ndarray, restored: np.ndarray) -> float:
    """
    計算峰值信噪比 (PSNR)。
    
    Args:
        original: 原始影像
        restored: 復原影像
        
    Returns:
        PSNR 值（dB）
    """
    original_float = original.astype(np.float64)
    restored_float = restored.astype(np.float64)
    
    mse = np.mean((original_float - restored_float) ** 2)
    if mse < 1e-10:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 10.0 * np.log10((max_pixel ** 2) / mse)
    return psnr


def process_image_pair(
    original_filename: str,
    degraded_filename: str,
    original_image: np.ndarray,
    degraded_image: np.ndarray,
    params: RestorationParameters,
    logger: logging.Logger,
    visualizer: ImageRestorationVisualizer,
    loader: ImageFileLoader,
    visualize: bool = True,
    save: bool = True,
) -> Tuple[RestorationResultsSchema, Dict[str, float]]:
    """
    處理一對原始與退化影像。
    
    Args:
        original_filename: 原始影像檔名
        degraded_filename: 退化影像檔名
        original_image: 原始影像陣列
        degraded_image: 退化影像陣列
        params: 復原參數
        logger: 日誌記錄器
        visualizer: 視覺化工具
        loader: 影像載入器
        visualize: 是否顯示視覺化結果
        save: 是否儲存結果
        
    Returns:
        (復原結果, 統計資訊) 元組
    """
    logger.info('Processing pair: %s <-> %s', original_filename, degraded_filename)
    
    results = compute_restoration_outputs(degraded_image, params, logger)
    stats = collect_intensity_statistics(results, original_image, degraded_image)
    
    # 計算 PSNR
    original_uint8 = sanitize_to_uint8(original_image)
    psnr_inverse = compute_psnr(original_uint8, results.inverse_filtered)
    psnr_wiener = compute_psnr(original_uint8, results.wiener_filtered)
    stats['psnr_inverse'] = psnr_inverse
    stats['psnr_wiener'] = psnr_wiener
    
    logger.info('  PSNR - Inverse: %.2f dB, Wiener: %.2f dB', psnr_inverse, psnr_wiener)
    
    if save:
        base_name = os.path.splitext(original_filename)[0]
        saved_paths = save_results(f'{base_name}_restored', results, loader)
        logger.info('  Saved: %s', list(saved_paths.values()))
    
    if visualize:
        visualize_results(
            image_filename=original_filename,
            original_image=original_image,
            degraded_image=degraded_image,
            results=results,
            visualizer=visualizer,
            display_plot_immediately=True,
        )
    
    return results, stats
