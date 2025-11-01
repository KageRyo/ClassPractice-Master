import os
import logging
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from src.schemas.enhancement_results_schema import EnhancementResultsSchema
from src.enhancement.power_law import apply_power_law_transformation
from src.enhancement.histogram_equalization import apply_histogram_equalization_enhancement
from src.enhancement.laplacian import apply_laplacian_image_sharpening
from src.ui.visualization import ImageEnhancementVisualizer
from src.utils.image_utils import ImageFileLoader, ImageHistogramCalculator


def compute_enhancements(image_array: np.ndarray, gamma_value: float, logger: logging.Logger) -> EnhancementResultsSchema:
    """Run all enhancement operations for a single image and return validated results."""
    logger.info("  Applying power-law transformation...")
    power_law_arr = apply_power_law_transformation(image_array, gamma_value=gamma_value)

    logger.info("  Applying histogram equalization...")
    hist_eq_arr = apply_histogram_equalization_enhancement(image_array)

    logger.info("  Applying Laplacian sharpening...")
    laplacian_arr = apply_laplacian_image_sharpening(image_array)

    return EnhancementResultsSchema(
        power_law=power_law_arr,
        hist_eq=hist_eq_arr,
        laplacian=laplacian_arr
    )


def visualize_results(image_filename: str,
                      original_image: np.ndarray,
                      results: EnhancementResultsSchema,
                      visualizer: ImageEnhancementVisualizer,
                      gamma_value: float,
                      figure_dir: str = 'results',
                      display_plot_immediately: bool = True) -> str:
    """Generate and (optionally) display comparison figure; return saved path."""
    base_name = os.path.splitext(image_filename)[0]
    os.makedirs(figure_dir, exist_ok=True)
    comparison_figure_path = os.path.join(figure_dir, f"{base_name}_comparison.png")
    visualizer.display_complete_enhancement_results(
        image_filename=image_filename,
        original_image_array=original_image,
        power_law_transformed_result=results.power_law,
        histogram_equalized_result=results.hist_eq,
        laplacian_sharpened_result=results.laplacian,
        gamma_value=gamma_value,
        figure_save_path=comparison_figure_path,
        display_plot_immediately=display_plot_immediately
    )
    return comparison_figure_path


def save_results(image_filename: str, results: EnhancementResultsSchema, loader: ImageFileLoader):
    """Persist each processed variant to disk using the naming convention from original code."""
    base_name = os.path.splitext(image_filename)[0]
    loader.save_image_array_to_file(results.power_law, f'{base_name}_gamma.bmp')
    loader.save_image_array_to_file(results.hist_eq, f'{base_name}_hist_eq.bmp')
    loader.save_image_array_to_file(results.laplacian, f'{base_name}_sharpened.bmp')


def save_histogram_figures(image_filename: str,
                           original_image: np.ndarray,
                           results: EnhancementResultsSchema,
                           output_root: str = 'results/histograms') -> Dict[str, str]:
    """Generate histogram plots for each variant and save to disk."""
    calculator = ImageHistogramCalculator()
    base_name = os.path.splitext(image_filename)[0]
    target_dir = os.path.join(output_root, base_name)
    os.makedirs(target_dir, exist_ok=True)

    variant_settings = [
        ('original', original_image, '#4c72b0'),
        ('power_law', results.power_law, '#55a868'),
        ('hist_eq', results.hist_eq, '#c44e52'),
        ('laplacian', results.laplacian, '#8172b3'),
    ]

    saved_paths: Dict[str, str] = {}
    for key, image_array, color in variant_settings:
        histogram = calculator.calculate_image_pixel_histogram(image_array)
        figure, axis = plt.subplots(figsize=(4, 2.2), dpi=120)
        axis.bar(range(256), histogram, color=color, alpha=0.85)
        axis.set_xlim(0, 255)
        axis.set_xlabel('Intensity', fontsize=8)
        axis.set_ylabel('Frequency', fontsize=8)
        axis.set_title(f'{key.replace("_", " ").title()} Histogram', fontsize=9)
        axis.tick_params(labelsize=8)
        figure.tight_layout()
        output_path = os.path.join(target_dir, f'{key}_hist.png')
        figure.savefig(output_path, bbox_inches='tight')
        plt.close(figure)
        saved_paths[key] = output_path

    return saved_paths


def process_single_image(image_filename: str,
                         image_array: np.ndarray,
                         gamma_value: float,
                         logger: logging.Logger,
                         visualizer: ImageEnhancementVisualizer,
                         loader: ImageFileLoader,
                         visualize: bool = True,
                         save: bool = True) -> EnhancementResultsSchema:
    """
    Full processing pipeline for one image (compute -> visualize -> save).
    Returns the validated enhancement results (useful for tests).
    """
    logger.info(f"Processing {image_filename}...")
    results = compute_enhancements(image_array, gamma_value, logger)
    if visualize:
        logger.info("  Displaying and saving results...")
        visualize_results(image_filename, image_array, results, visualizer, gamma_value, display_plot_immediately=True)
    if save:
        save_results(image_filename, results, loader)
    logger.info(f"  Processing completed for {image_filename}")
    return results
