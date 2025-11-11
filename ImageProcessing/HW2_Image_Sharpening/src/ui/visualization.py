import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.schemas.sharpening_results_schema import SharpeningResultsSchema

logger = logging.getLogger(__name__)


class ImageSharpeningVisualizer:
    """Create comparison figures for original vs. sharpening results."""

    def __init__(self, figure_size=(18, 4.5), dpi=180):
        self.figure_size = figure_size
        self.dpi = dpi
        logger.info('ImageSharpeningVisualizer initialized (size=%s, dpi=%d)', figure_size, dpi)

    def display_sharpening_results(
        self,
        image_filename: str,
        original_image: np.ndarray,
        results: SharpeningResultsSchema,
        figure_save_path: Optional[str] = None,
        display_plot_immediately: bool = True,
    ) -> Optional[str]:
        figure, axes = plt.subplots(1, 5, figsize=self.figure_size, dpi=self.dpi)
        figure.suptitle(f'Image Sharpening Results - {image_filename}', fontsize=16)

        images = [
            (original_image, 'Original'),
            (results.laplacian, 'Laplacian'),
            (results.unsharp_mask, 'Unsharp Mask'),
            (results.high_boost, 'High-Boost'),
            (results.homomorphic, 'Homomorphic'),
        ]

        for axis, (image_array, title) in zip(axes, images):
            axis.imshow(image_array, cmap='gray', vmin=0, vmax=255)
            axis.set_title(title, fontsize=12)
            axis.axis('off')

        figure.tight_layout()
        figure.subplots_adjust(top=0.85)

        if figure_save_path:
            figure.savefig(figure_save_path, bbox_inches='tight', pad_inches=0.2)
            logger.debug('Sharpening comparison figure saved to %s', figure_save_path)

        if display_plot_immediately:
            plt.show()
        else:
            plt.close(figure)

        return figure_save_path
