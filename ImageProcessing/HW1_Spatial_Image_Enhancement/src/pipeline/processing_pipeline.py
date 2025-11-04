import os
import logging
from typing import Dict, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from src.schemas.enhancement_results_schema import EnhancementResultsSchema
from src.enhancement.power_law import apply_power_law_transformation, estimate_gamma_for_brightness
from src.enhancement.histogram_equalization import apply_histogram_equalization_enhancement
from src.enhancement.laplacian import apply_laplacian_image_sharpening
from src.ui.visualization import ImageEnhancementVisualizer
from src.utils.image_utils import ImageFileLoader, ImageHistogramCalculator


def compute_enhancements(image_array: np.ndarray, gamma_value: float, logger: logging.Logger) -> EnhancementResultsSchema:
    """Run enhancement operations and synthesize the combined gamma-then-Laplacian output."""
    logger.info(f"  Applying power-law transformation (gamma={gamma_value:.3f})...")
    power_law_image = apply_power_law_transformation(image_array, gamma_value=gamma_value)

    logger.info("  Applying histogram equalization...")
    histogram_equalized_image = apply_histogram_equalization_enhancement(image_array)

    logger.info("  Applying Laplacian sharpening on gamma-corrected image...")
    gamma_then_laplacian_image = apply_laplacian_image_sharpening(power_law_image)

    return EnhancementResultsSchema(
        power_law=power_law_image,
        hist_eq=histogram_equalized_image,
        gamma_laplacian=gamma_then_laplacian_image
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
        gamma_then_laplacian_result=results.gamma_laplacian,
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
    loader.save_image_array_to_file(results.gamma_laplacian, f'{base_name}_gamma_laplacian.bmp')


def save_histogram_figures(image_filename: str,
                           original_image: np.ndarray,
                           results: EnhancementResultsSchema,
                           output_root: str = 'results') -> Dict[str, str]:
    """Generate histogram plots for each variant and save to disk."""
    calculator = ImageHistogramCalculator()
    base_name = os.path.splitext(image_filename)[0]
    os.makedirs(output_root, exist_ok=True)

    variant_settings = [
        ('original', original_image, '#4c72b0'),
        ('power_law', results.power_law, '#55a868'),
        ('hist_eq', results.hist_eq, '#c44e52'),
    ('gamma_laplacian', results.gamma_laplacian, '#8172b3'),
    ]

    saved_paths: Dict[str, str] = {}
    for key, image_array, color in variant_settings:
        histogram = calculator.calculate_image_pixel_histogram(image_array)
        figure, axis = plt.subplots(figsize=(5.0, 3.0), dpi=120)
        axis.bar(range(256), histogram, color=color, alpha=0.85, width=1.0, align='edge')
        axis.set_xlim(-0.5, 255.5)
        max_count = max(histogram) if histogram else 0
        if max_count > 0:
            axis.set_ylim(0, max_count * 1.05)
        axis.set_xlabel('Intensity', fontsize=8)
        axis.set_ylabel('Frequency', fontsize=8)
        axis.set_title(f'{key.replace("_", " ").title()} Histogram', fontsize=9)
        axis.tick_params(labelsize=8)
        figure.tight_layout()
        output_path = os.path.join(output_root, f'{base_name}_{key}_hist.png')
        figure.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(figure)
        saved_paths[key] = output_path

    return saved_paths



def process_single_image(image_filename: str,
                         image_array: np.ndarray,
                         gamma_value: Optional[float],
                         logger: logging.Logger,
                         visualizer: ImageEnhancementVisualizer,
                         loader: ImageFileLoader,
                         visualize: bool = True,
                         save: bool = True,
                         auto_gamma_bounds: Tuple[float, float] = (0.35, 2.0),
                         target_mean_intensity: float = 0.6) -> Tuple[EnhancementResultsSchema, float]:
    """
    Full processing pipeline for one image (compute -> visualize -> save).
    Returns the validated enhancement results (useful for tests).
    """
    logger.info(f"Processing {image_filename}...")
    lower_bound, upper_bound = auto_gamma_bounds if auto_gamma_bounds[0] <= auto_gamma_bounds[1] else (auto_gamma_bounds[1], auto_gamma_bounds[0])

    resolved_gamma = gamma_value
    if resolved_gamma is None:
        resolved_gamma = estimate_gamma_for_brightness(
            image_array,
            target_mean=target_mean_intensity,
            min_gamma=lower_bound,
            max_gamma=upper_bound,
        )
        logger.info(
            "  Auto-selected gamma %.3f to target mean %.2f",
            resolved_gamma,
            target_mean_intensity,
        )
    else:
        if resolved_gamma < lower_bound or resolved_gamma > upper_bound:
            clamped_gamma = float(np.clip(resolved_gamma, lower_bound, upper_bound))
            logger.warning(
                "  Clamping provided gamma %.3f to %.3f (bounds %.2f-%.2f)",
                resolved_gamma,
                clamped_gamma,
                lower_bound,
                upper_bound,
            )
            resolved_gamma = clamped_gamma
        else:
            resolved_gamma = float(resolved_gamma)
        logger.info("  Using provided gamma %.3f", resolved_gamma)

    results = compute_enhancements(image_array, resolved_gamma, logger)
    if visualize:
        logger.info("  Displaying and saving results...")
        visualize_results(image_filename, image_array, results, visualizer, resolved_gamma, display_plot_immediately=True)
    if save:
        save_results(image_filename, results, loader)
    logger.info(f"  Processing completed for {image_filename}")
    return results, resolved_gamma
