import matplotlib.pyplot as plt
import logging
from ..utils.image_utils import ImageHistogramCalculator

logger = logging.getLogger(__name__)

class ImageEnhancementVisualizer:
    """Render images and their histograms for comparison and analysis."""

    def __init__(self, figure_size_dimensions=(16, 8), image_resolution_dpi=300):
        """Configure figure size and export resolution (dpi)."""
        self.figure_size_dimensions = figure_size_dimensions
        self.image_resolution_dpi = image_resolution_dpi
        self.histogram_calculator = ImageHistogramCalculator()
        logger.info(f"ImageEnhancementVisualizer initialized with figure_size_dimensions={figure_size_dimensions}, image_resolution_dpi={image_resolution_dpi}")

    def display_complete_enhancement_results(self, image_filename, original_image_array,
                                             power_law_transformed_result, histogram_equalized_result,
                                             laplacian_sharpened_result, gamma_value=2.2,
                                             figure_save_path=None, display_plot_immediately=True):
        """Display four processed images and their histograms in a 2x4 grid."""
        matplotlib_figure, subplot_axes = plt.subplots(2, 4, figsize=self.figure_size_dimensions)
        matplotlib_figure.suptitle(f'Image Enhancement Results - {image_filename}', fontsize=16)
        self._display_single_image_on_axes(subplot_axes[0, 0], original_image_array, 'Original Image')
        self._display_single_image_on_axes(subplot_axes[0, 1], power_law_transformed_result, f'Power-law (γ={gamma_value:.3f})')
        self._display_single_image_on_axes(subplot_axes[0, 2], histogram_equalized_result, 'Histogram Equalization')
        self._display_single_image_on_axes(subplot_axes[0, 3], laplacian_sharpened_result, 'Laplacian Sharpening')

        processed_image_list = [original_image_array, power_law_transformed_result, histogram_equalized_result, laplacian_sharpened_result]
        histogram_title_list = ['Original Histogram', 'Power-law Histogram',
                                'Equalized Histogram', 'Sharpened Histogram']
        histogram_color_list = ['blue', 'green', 'red', 'orange']
        for i, (processed_image, histogram_title, histogram_color) in enumerate(zip(processed_image_list, histogram_title_list, histogram_color_list)):
            self._display_histogram_on_axes(subplot_axes[1, i], processed_image, histogram_title, histogram_color)
        plt.tight_layout()
        if figure_save_path:
            plt.savefig(figure_save_path, dpi=self.image_resolution_dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {figure_save_path}")
        if display_plot_immediately:
            plt.show()
        else:
            plt.close()

    def _display_single_image_on_axes(self, matplotlib_axes, image_array, image_title):
        """Render one grayscale image without axes."""
        matplotlib_axes.imshow(image_array, cmap='gray', vmin=0, vmax=255)
        matplotlib_axes.set_title(image_title)
        matplotlib_axes.axis('off')

    def _display_histogram_on_axes(self, matplotlib_axes, image_array, histogram_title, bar_color):
        """Plot intensity histogram for given image array."""
        pixel_intensity_histogram = self.histogram_calculator.calculate_image_pixel_histogram(image_array)
        intensity_value_range = list(range(256))
        matplotlib_axes.bar(intensity_value_range, pixel_intensity_histogram, alpha=0.7, color=bar_color)
        matplotlib_axes.set_title(histogram_title)
        matplotlib_axes.set_xlabel('Pixel Intensity')
        matplotlib_axes.set_ylabel('Frequency')
        matplotlib_axes.set_xlim(0, 255)

    def display_single_image_with_histogram_analysis(self, image_array, image_title, figure_save_path=None, display_plot_immediately=True):
        """Display one image and its histogram side by side."""
        matplotlib_figure, (image_axes, histogram_axes) = plt.subplots(1, 2, figsize=(12, 4))
        matplotlib_figure.suptitle(image_title, fontsize=14)
        self._display_single_image_on_axes(image_axes, image_array, 'Image')
        self._display_histogram_on_axes(histogram_axes, image_array, 'Histogram', 'blue')
        plt.tight_layout()
        if figure_save_path:
            plt.savefig(figure_save_path, dpi=self.image_resolution_dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {figure_save_path}")
        if display_plot_immediately:
            plt.show()
        else:
            plt.close()