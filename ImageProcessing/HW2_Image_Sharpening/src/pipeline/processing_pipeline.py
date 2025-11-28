import logging
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.schemas.sharpening_results_schema import SharpeningResultsSchema
from src.sharpening.laplacian import apply_laplacian_sharpening
from src.sharpening.unsharp_mask import apply_unsharp_masking
from src.sharpening.high_boost import apply_high_boost_filter
from src.sharpening.homomorphic import apply_homomorphic_filter
from src.ui.visualization import ImageSharpeningVisualizer
from src.utils.image_utils import ImageFileLoader


@dataclass
class SharpeningParameters:
    laplacian_kernel: str = '8-connected'
    laplacian_alpha: float = 1.0
    unsharp_amount: float = 1.0
    high_boost_factor: float = 1.8
    homomorphic_gamma_l: float = 0.8
    homomorphic_gamma_h: float = 1.5
    homomorphic_cutoff: float = 40.0
    homomorphic_c: float = 1.2

    def summarize(self) -> str:
        return (
            f"Laplacian: {self.laplacian_kernel} (alpha={self.laplacian_alpha:.2f}) | "
            f"Unsharp amount={self.unsharp_amount:.2f} | "
            f"High-boost factor={self.high_boost_factor:.2f} | "
            f"Homomorphic gamma_l={self.homomorphic_gamma_l:.2f}, gamma_h={self.homomorphic_gamma_h:.2f}, "
            f"D0={self.homomorphic_cutoff:.1f}, c={self.homomorphic_c:.2f}"
        )


def sanitize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float input to uint8 safely."""
    return np.clip(np.rint(np.asarray(image, dtype=np.float64)), 0, 255).astype(np.uint8)


def compute_sharpening_outputs(
    image_array: np.ndarray,
    params: SharpeningParameters,
    logger: logging.Logger,
) -> SharpeningResultsSchema:
    image_uint8 = sanitize_to_uint8(image_array)

    logger.info('  Applying Laplacian sharpening (%s kernel)...',
                params.laplacian_kernel)
    laplacian_image = apply_laplacian_sharpening(
        image_uint8,
        kernel_type=params.laplacian_kernel,
        alpha=params.laplacian_alpha,
    )

    logger.info('  Applying unsharp masking (amount=%.2f)...',
                params.unsharp_amount)
    unsharp_image = apply_unsharp_masking(
        image_uint8, amount=params.unsharp_amount)

    logger.info('  Applying high-boost filtering (factor=%.2f)...',
                params.high_boost_factor)
    high_boost_image = apply_high_boost_filter(
        image_uint8, boost_factor=params.high_boost_factor)

    logger.info(
        '  Applying homomorphic filtering (gamma_l=%.2f, gamma_h=%.2f, D0=%.1f, c=%.2f)...',
        params.homomorphic_gamma_l,
        params.homomorphic_gamma_h,
        params.homomorphic_cutoff,
        params.homomorphic_c,
    )
    homomorphic_image = apply_homomorphic_filter(
        image_uint8,
        gamma_l=params.homomorphic_gamma_l,
        gamma_h=params.homomorphic_gamma_h,
        cutoff_frequency=params.homomorphic_cutoff,
        c=params.homomorphic_c,
    )

    results = SharpeningResultsSchema(
        laplacian=laplacian_image,
        unsharp_mask=unsharp_image,
        high_boost=high_boost_image,
        homomorphic=homomorphic_image,
    )
    return results


def visualize_results(
    image_filename: str,
    original_image: np.ndarray,
    results: SharpeningResultsSchema,
    visualizer: ImageSharpeningVisualizer,
    figure_dir: str = 'results',
    display_plot_immediately: bool = True,
) -> str:
    base_name = os.path.splitext(image_filename)[0]
    os.makedirs(figure_dir, exist_ok=True)
    comparison_figure_path = os.path.join(
        figure_dir, f'{base_name}_sharpening_comparison.png')
    visualizer.display_sharpening_results(
        image_filename=image_filename,
        original_image=sanitize_to_uint8(original_image),
        results=results,
        figure_save_path=comparison_figure_path,
        display_plot_immediately=display_plot_immediately,
    )
    return comparison_figure_path


def save_results(
    image_filename: str,
    results: SharpeningResultsSchema,
    loader: ImageFileLoader,
) -> Dict[str, str]:
    base_name = os.path.splitext(image_filename)[0]
    saved_paths = {
        'laplacian': loader.save_image_array_to_file(results.laplacian, f'{base_name}_laplacian.bmp'),
        'unsharp_mask': loader.save_image_array_to_file(results.unsharp_mask, f'{base_name}_unsharp.bmp'),
        'high_boost': loader.save_image_array_to_file(results.high_boost, f'{base_name}_high_boost.bmp'),
        'homomorphic': loader.save_image_array_to_file(results.homomorphic, f'{base_name}_homomorphic.bmp'),
    }
    return saved_paths


def collect_intensity_statistics(results: SharpeningResultsSchema, original_image: np.ndarray) -> Dict[str, float]:
    def mean_intensity(image: np.ndarray) -> float:
        return float(np.mean(image))

    stats = {
        'original_mean': mean_intensity(sanitize_to_uint8(original_image)),
        'laplacian_mean': mean_intensity(results.laplacian),
        'unsharp_mean': mean_intensity(results.unsharp_mask),
        'high_boost_mean': mean_intensity(results.high_boost),
        'homomorphic_mean': mean_intensity(results.homomorphic),
    }
    return stats


def process_single_image(
    image_filename: str,
    image_array: np.ndarray,
    params: SharpeningParameters,
    logger: logging.Logger,
    visualizer: ImageSharpeningVisualizer,
    loader: ImageFileLoader,
    visualize: bool = True,
    save: bool = True,
) -> Tuple[SharpeningResultsSchema, Dict[str, float]]:
    logger.info('Processing %s...', image_filename)
    results = compute_sharpening_outputs(image_array, params, logger)
    if visualize:
        logger.info('  Generating comparison figure...')
        visualize_results(
            image_filename=image_filename,
            original_image=image_array,
            results=results,
            visualizer=visualizer,
            display_plot_immediately=False,
        )
    if save:
        logger.info('  Saving sharpened outputs...')
        save_paths = save_results(image_filename, results, loader)
        for key, path in save_paths.items():
            logger.debug('    Saved %s to %s', key, path)
    stats = collect_intensity_statistics(results, image_array)
    logger.info('  Completed %s', image_filename)
    return results, stats
