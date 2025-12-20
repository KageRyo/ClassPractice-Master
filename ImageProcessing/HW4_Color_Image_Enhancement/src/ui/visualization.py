"""
Color Image Enhancement Visualization Module

Provides visualization utilities for displaying color images and their histograms.
"""

import matplotlib.pyplot as plt
import logging
from ..utils.image_utils import ColorHistogramCalculator

logger = logging.getLogger(__name__)


class ColorEnhancementVisualizer:
    """Render color images and their histograms for comparison and analysis."""

    def __init__(self, figure_size_dimensions=(18, 10), image_resolution_dpi=300):
        """Configure figure size and export resolution (dpi)."""
        self.figure_size_dimensions = figure_size_dimensions
        self.image_resolution_dpi = image_resolution_dpi
        self.histogram_calculator = ColorHistogramCalculator()
        logger.info(f"ColorEnhancementVisualizer initialized with figure_size={figure_size_dimensions}, dpi={image_resolution_dpi}")

    def display_complete_enhancement_results(self, image_filename, original_image,
                                             rgb_hist_eq_result, hsi_hist_eq_result,
                                             hsi_gamma_result, hsi_saturation_result,
                                             gamma_value=1.0,
                                             figure_save_path=None, 
                                             display_plot_immediately=True):
        """Display five color images (original + 4 enhanced) and their intensity histograms in a 2x5 grid."""
        matplotlib_figure, subplot_axes = plt.subplots(2, 5, figsize=self.figure_size_dimensions)
        matplotlib_figure.suptitle(f'Color Image Enhancement Results - {image_filename}', fontsize=16)
        
        # Row 1: Images
        self.display_color_image_on_axes(subplot_axes[0, 0], original_image, 'Original Image')
        self.display_color_image_on_axes(subplot_axes[0, 1], rgb_hist_eq_result, 'RGB Hist. Eq.')
        self.display_color_image_on_axes(subplot_axes[0, 2], hsi_hist_eq_result, 'HSI Intensity Hist. Eq.')
        self.display_color_image_on_axes(subplot_axes[0, 3], hsi_gamma_result, f'HSI Gamma (γ={gamma_value:.2f})')
        self.display_color_image_on_axes(subplot_axes[0, 4], hsi_saturation_result, 'HSI Saturation Enh.')

        # Row 2: Intensity Histograms
        images_list = [original_image, rgb_hist_eq_result, hsi_hist_eq_result, hsi_gamma_result, hsi_saturation_result]
        histogram_titles = ['Original Histogram', 'RGB Hist. Eq. Histogram', 
                           'HSI Hist. Eq. Histogram', 'HSI Gamma Histogram', 'HSI Saturation Histogram']
        histogram_colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i, (image, title, color) in enumerate(zip(images_list, histogram_titles, histogram_colors)):
            self.display_intensity_histogram_on_axes(subplot_axes[1, i], image, title, color)

        plt.tight_layout()
        
        if figure_save_path:
            plt.savefig(figure_save_path, dpi=self.image_resolution_dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {figure_save_path}")
        
        if display_plot_immediately:
            plt.show()
        else:
            plt.close()

    def display_color_image_on_axes(self, matplotlib_axes, image_array, image_title):
        """Render one color image without axes."""
        # Ensure uint8 for display
        if image_array.dtype != 'uint8':
            display_image = image_array.astype('uint8')
        else:
            display_image = image_array
        matplotlib_axes.imshow(display_image)
        matplotlib_axes.set_title(image_title, fontsize=10)
        matplotlib_axes.axis('off')

    def display_intensity_histogram_on_axes(self, matplotlib_axes, color_image, histogram_title, bar_color):
        """Plot intensity histogram for given color image."""
        intensity_histogram = self.histogram_calculator.calculate_intensity_histogram(color_image)
        intensity_range = list(range(256))
        matplotlib_axes.bar(intensity_range, intensity_histogram, alpha=0.7, color=bar_color, width=1.0)
        matplotlib_axes.set_title(histogram_title, fontsize=9)
        matplotlib_axes.set_xlabel('Intensity', fontsize=8)
        matplotlib_axes.set_ylabel('Frequency', fontsize=8)
        matplotlib_axes.set_xlim(0, 255)
        matplotlib_axes.tick_params(labelsize=7)

    def display_rgb_histograms_on_axes(self, matplotlib_axes, color_image, title_prefix):
        """Plot R, G, B channel histograms overlaid."""
        rgb_histograms = self.histogram_calculator.calculate_rgb_histograms(color_image)
        intensity_range = list(range(256))
        
        matplotlib_axes.bar(intensity_range, rgb_histograms['R'], alpha=0.5, color='red', width=1.0, label='R')
        matplotlib_axes.bar(intensity_range, rgb_histograms['G'], alpha=0.5, color='green', width=1.0, label='G')
        matplotlib_axes.bar(intensity_range, rgb_histograms['B'], alpha=0.5, color='blue', width=1.0, label='B')
        
        matplotlib_axes.set_title(f'{title_prefix} RGB Histogram', fontsize=9)
        matplotlib_axes.set_xlabel('Intensity', fontsize=8)
        matplotlib_axes.set_ylabel('Frequency', fontsize=8)
        matplotlib_axes.set_xlim(0, 255)
        matplotlib_axes.legend(fontsize=7)
        matplotlib_axes.tick_params(labelsize=7)

    def display_single_image_with_histogram(self, image_array, image_title, 
                                            figure_save_path=None, display_plot_immediately=True):
        """Display one color image and its intensity histogram side by side."""
        matplotlib_figure, (image_axes, histogram_axes) = plt.subplots(1, 2, figsize=(12, 5))
        matplotlib_figure.suptitle(image_title, fontsize=14)
        
        self.display_color_image_on_axes(image_axes, image_array, 'Image')
        self.display_intensity_histogram_on_axes(histogram_axes, image_array, 'Intensity Histogram', 'blue')
        
        plt.tight_layout()
        
        if figure_save_path:
            plt.savefig(figure_save_path, dpi=self.image_resolution_dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {figure_save_path}")
        
        if display_plot_immediately:
            plt.show()
        else:
            plt.close()

    def display_comparison_with_rgb_histograms(self, image_filename, original_image,
                                               enhanced_images_dict, gamma_value=1.0,
                                               figure_save_path=None, display_plot_immediately=True):
        """
        Display comparison with RGB histograms instead of intensity histograms.
        
        Args:
            enhanced_images_dict: Dict with keys as technique names and values as enhanced images
        """
        n_images = 1 + len(enhanced_images_dict)
        fig, axes = plt.subplots(2, n_images, figsize=(4 * n_images, 8))
        fig.suptitle(f'Color Enhancement Comparison - {image_filename}', fontsize=16)
        
        # Original
        self.display_color_image_on_axes(axes[0, 0], original_image, 'Original')
        self.display_rgb_histograms_on_axes(axes[1, 0], original_image, 'Original')
        
        # Enhanced images
        for idx, (technique_name, enhanced_image) in enumerate(enhanced_images_dict.items(), 1):
            self.display_color_image_on_axes(axes[0, idx], enhanced_image, technique_name)
            self.display_rgb_histograms_on_axes(axes[1, idx], enhanced_image, technique_name)
        
        plt.tight_layout()
        
        if figure_save_path:
            plt.savefig(figure_save_path, dpi=self.image_resolution_dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {figure_save_path}")
        
        if display_plot_immediately:
            plt.show()
        else:
            plt.close()
