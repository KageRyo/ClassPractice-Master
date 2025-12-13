import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.schemas.restoration_results_schema import RestorationResultsSchema

logger = logging.getLogger(__name__)


class ImageRestorationVisualizer:
    """Create comparison figures for original, degraded, and restored images."""

    def __init__(self, figure_size=(20, 5), dpi=180):
        self.figure_size = figure_size
        self.dpi = dpi
        logger.info(
            'ImageRestorationVisualizer initialized (size=%s, dpi=%d)', figure_size, dpi)

    def display_restoration_results(
        self,
        image_filename: str,
        original_image: np.ndarray,
        degraded_image: np.ndarray,
        results: RestorationResultsSchema,
        figure_save_path: Optional[str] = None,
        display_plot_immediately: bool = True,
    ) -> Optional[str]:
        """
        顯示原始、退化和復原影像的比較圖（共4張圖）。
        
        Args:
            image_filename: 影像檔名
            original_image: 原始影像
            degraded_image: 退化影像
            results: 復原結果
            figure_save_path: 圖形儲存路徑
            display_plot_immediately: 是否立即顯示
            
        Returns:
            儲存路徑（如果有儲存）
        """
        figure, axes = plt.subplots(
            1, 4, figsize=self.figure_size, dpi=self.dpi)
        figure.suptitle(
            f'Image Restoration Results - {image_filename}', fontsize=16)

        images = [
            (original_image, 'Original'),
            (degraded_image, 'Degraded'),
            (results.inverse_filtered, 'Inverse Filter'),
            (results.wiener_filtered, 'Wiener Filter'),
        ]

        for axis, (image_array, title) in zip(axes, images):
            axis.imshow(image_array, cmap='gray', vmin=0, vmax=255)
            axis.set_title(title, fontsize=12)
            axis.axis('off')

        figure.tight_layout()
        figure.subplots_adjust(top=0.85)

        if figure_save_path:
            figure.savefig(figure_save_path,
                           bbox_inches='tight', pad_inches=0.2)
            logger.debug(
                'Restoration comparison figure saved to %s', figure_save_path)

        if display_plot_immediately:
            plt.show()
        else:
            plt.close(figure)

        return figure_save_path

    def display_frequency_spectrum(
        self,
        image: np.ndarray,
        title: str = 'Frequency Spectrum',
        figure_save_path: Optional[str] = None,
        display_plot_immediately: bool = True,
    ) -> Optional[str]:
        """
        顯示影像的頻率譜。
        
        Args:
            image: 灰階影像
            title: 圖形標題
            figure_save_path: 儲存路徑
            display_plot_immediately: 是否立即顯示
            
        Returns:
            儲存路徑
        """
        # 計算 FFT
        frequency = np.fft.fft2(image.astype(np.float64))
        frequency_shifted = np.fft.fftshift(frequency)
        magnitude = np.log(1 + np.abs(frequency_shifted))
        
        figure, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)
        ax.imshow(magnitude, cmap='gray')
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        figure.tight_layout()
        
        if figure_save_path:
            figure.savefig(figure_save_path,
                           bbox_inches='tight', pad_inches=0.2)
            logger.debug('Frequency spectrum saved to %s', figure_save_path)
        
        if display_plot_immediately:
            plt.show()
        else:
            plt.close(figure)
        
        return figure_save_path
