"""
Color Image Enhancement Processing Pipeline

This module provides the main processing pipeline for color image enhancement.
It orchestrates the conversion between color spaces and applies various enhancement techniques.
"""

import os
import logging
from typing import Dict, Optional, Tuple

import numpy as np
from src.schemas.enhancement_results_schema import ColorEnhancementResultsSchema
from src.color_space.color_conversion import rgb_to_hsi, hsi_to_rgb
from src.enhancement.histogram_equalization import (
    apply_rgb_histogram_equalization,
    apply_hsi_intensity_histogram_equalization
)
from src.enhancement.gamma_correction import (
    apply_rgb_gamma_correction,
    apply_hsi_intensity_gamma_correction,
    estimate_gamma_for_color_brightness
)
from src.enhancement.saturation_enhancement import apply_saturation_enhancement
from src.enhancement.intensity_enhancement import (
    apply_intensity_contrast_stretching,
    apply_rgb_contrast_stretching
)
from src.ui.visualization import ColorEnhancementVisualizer
from src.utils.image_utils import ColorImageFileLoader


class ColorEnhancementPipeline:
    """Main pipeline for processing color images with various enhancement techniques."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def enhance_in_rgb_space(self, rgb_image, technique='histogram_eq', **kwargs):
        """
        Apply enhancement directly in RGB color space.
        
        Args:
            rgb_image: Input RGB image array
            technique: Enhancement technique ('histogram_eq', 'gamma', 'contrast_stretch')
            **kwargs: Additional parameters for the technique
            
        Returns:
            Enhanced RGB image
        """
        self.logger.info(f"Applying RGB-space enhancement: {technique}")
        
        if technique == 'histogram_eq':
            return apply_rgb_histogram_equalization(rgb_image)
        elif technique == 'gamma':
            gamma = kwargs.get('gamma', 1.0)
            return apply_rgb_gamma_correction(rgb_image, gamma)
        elif technique == 'contrast_stretch':
            low = kwargs.get('low_percentile', 1)
            high = kwargs.get('high_percentile', 99)
            return apply_rgb_contrast_stretching(rgb_image, low, high)
        else:
            self.logger.warning(f"Unknown technique: {technique}, returning original")
            return rgb_image

    def enhance_in_hsi_space(self, rgb_image, technique='intensity_histogram_eq', **kwargs):
        """
        Apply enhancement in HSI color space (preserving Hue).
        
        Args:
            rgb_image: Input RGB image array
            technique: Enhancement technique 
                      ('intensity_histogram_eq', 'intensity_gamma', 
                       'saturation', 'intensity_contrast_stretch')
            **kwargs: Additional parameters for the technique
            
        Returns:
            Enhanced RGB image (converted back from HSI)
        """
        self.logger.info(f"Applying HSI-space enhancement: {technique}")
        
        # Convert to HSI
        hsi_image = rgb_to_hsi(rgb_image)
        
        # Apply enhancement
        if technique == 'intensity_histogram_eq':
            enhanced_hsi = apply_hsi_intensity_histogram_equalization(hsi_image)
        elif technique == 'intensity_gamma':
            gamma = kwargs.get('gamma', 1.0)
            enhanced_hsi = apply_hsi_intensity_gamma_correction(hsi_image, gamma)
        elif technique == 'saturation':
            factor = kwargs.get('factor', 1.5)
            enhanced_hsi = apply_saturation_enhancement(hsi_image, factor)
        elif technique == 'intensity_contrast_stretch':
            low = kwargs.get('low_percentile', 1)
            high = kwargs.get('high_percentile', 99)
            enhanced_hsi = apply_intensity_contrast_stretching(hsi_image, low, high)
        elif technique == 'combined':
            # Combined enhancement: gamma + saturation
            gamma = kwargs.get('gamma', 1.0)
            sat_factor = kwargs.get('saturation_factor', 1.2)
            enhanced_hsi = apply_hsi_intensity_gamma_correction(hsi_image, gamma)
            enhanced_hsi = apply_saturation_enhancement(enhanced_hsi, sat_factor)
        else:
            self.logger.warning(f"Unknown technique: {technique}, returning original")
            enhanced_hsi = hsi_image
        
        # Convert back to RGB
        enhanced_rgb = hsi_to_rgb(enhanced_hsi)
        return enhanced_rgb


def compute_color_enhancements(rgb_image, gamma_value, logger):
    """
    Run multiple enhancement operations on a color image.
    
    Args:
        rgb_image: Original RGB image array
        gamma_value: Gamma value for gamma correction
        logger: Logger instance
        
    Returns:
        ColorEnhancementResultsSchema with all enhancement results
    """
    pipeline = ColorEnhancementPipeline(logger)
    
    logger.info(f"  Applying RGB histogram equalization...")
    rgb_hist_eq = pipeline.enhance_in_rgb_space(rgb_image, 'histogram_eq')
    
    logger.info(f"  Applying HSI intensity histogram equalization...")
    hsi_hist_eq = pipeline.enhance_in_hsi_space(rgb_image, 'intensity_histogram_eq')
    
    logger.info(f"  Applying HSI intensity contrast stretching...")
    hsi_intensity_contrast = pipeline.enhance_in_hsi_space(
        rgb_image, 'intensity_contrast_stretch', low_percentile=1, high_percentile=99
    )

    logger.info(f"  Applying HSI intensity gamma correction (gamma={gamma_value:.3f})...")
    hsi_gamma = pipeline.enhance_in_hsi_space(rgb_image, 'intensity_gamma', gamma=gamma_value)
    
    logger.info(f"  Applying HSI saturation enhancement...")
    hsi_saturation = pipeline.enhance_in_hsi_space(rgb_image, 'saturation', factor=1.4)
    
    return ColorEnhancementResultsSchema(
        rgb_histogram_eq=rgb_hist_eq,
        hsi_histogram_eq=hsi_hist_eq,
        hsi_intensity_contrast=hsi_intensity_contrast,
        hsi_gamma=hsi_gamma,
        hsi_saturation=hsi_saturation
    )


def visualize_color_results(image_filename: str,
                            original_image: np.ndarray,
                            results: ColorEnhancementResultsSchema,
                            visualizer: ColorEnhancementVisualizer,
                            gamma_value: float,
                            figure_dir: str = 'results',
                            display_plot_immediately: bool = True) -> str:
    """Generate and (optionally) display comparison figure; return saved path."""
    base_name = os.path.splitext(image_filename)[0]
    os.makedirs(figure_dir, exist_ok=True)
    comparison_figure_path = os.path.join(figure_dir, f"{base_name}_comparison.png")
    
    visualizer.display_complete_enhancement_results(
        image_filename=image_filename,
        original_image=original_image,
        rgb_hist_eq_result=results.rgb_histogram_eq,
        hsi_hist_eq_result=results.hsi_histogram_eq,
        hsi_intensity_contrast_result=results.hsi_intensity_contrast,
        hsi_gamma_result=results.hsi_gamma,
        hsi_saturation_result=results.hsi_saturation,
        gamma_value=gamma_value,
        figure_save_path=comparison_figure_path,
        display_plot_immediately=display_plot_immediately
    )
    return comparison_figure_path


def save_color_results(image_filename: str, 
                       results: ColorEnhancementResultsSchema, 
                       loader: ColorImageFileLoader):
    """Persist each processed variant to disk."""
    base_name = os.path.splitext(image_filename)[0]
    loader.save_color_image_array(results.rgb_histogram_eq, f'{base_name}_rgb_hist_eq.png')
    loader.save_color_image_array(results.hsi_histogram_eq, f'{base_name}_hsi_hist_eq.png')
    loader.save_color_image_array(
        results.hsi_intensity_contrast, f'{base_name}_hsi_intensity_contrast_stretch.png'
    )
    loader.save_color_image_array(results.hsi_gamma, f'{base_name}_hsi_gamma.png')
    loader.save_color_image_array(results.hsi_saturation, f'{base_name}_hsi_saturation.png')


def process_single_color_image(image_filename: str,
                               image_array: np.ndarray,
                               gamma_value: Optional[float],
                               logger,
                               visualizer: ColorEnhancementVisualizer,
                               loader: ColorImageFileLoader,
                               visualize: bool = True,
                               save: bool = True,
                               auto_gamma_bounds: Tuple[float, float] = (0.3, 2.5),
                               target_mean_intensity: float = 0.5) -> Tuple[ColorEnhancementResultsSchema, float, str]:
    """
    Full processing pipeline for one color image.
    
    Returns:
        Tuple of (results, resolved_gamma, comparison_figure_path)
    """
    logger.info(f"Processing color image: {image_filename}...")
    
    lower_bound, upper_bound = auto_gamma_bounds
    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound
    
    resolved_gamma = gamma_value
    if resolved_gamma is None:
        resolved_gamma = estimate_gamma_for_color_brightness(
            image_array,
            target_mean=target_mean_intensity,
            min_gamma=lower_bound,
            max_gamma=upper_bound
        )
        logger.info(f"  Auto-selected gamma {resolved_gamma:.3f} for target mean {target_mean_intensity}")
    else:
        if resolved_gamma < lower_bound or resolved_gamma > upper_bound:
            clamped_gamma = float(np.clip(resolved_gamma, lower_bound, upper_bound))
            logger.warning(f"  Clamping provided gamma {resolved_gamma:.3f} to {clamped_gamma:.3f}")
            resolved_gamma = clamped_gamma
        logger.info(f"  Using provided gamma {resolved_gamma:.3f}")
    
    results = compute_color_enhancements(image_array, resolved_gamma, logger)
    
    comparison_figure_path = None
    if visualize:
        comparison_figure_path = visualize_color_results(
            image_filename, image_array, results, visualizer, resolved_gamma,
            display_plot_immediately=False
        )
        logger.info(f"  Saved comparison figure: {comparison_figure_path}")
    
    if save:
        save_color_results(image_filename, results, loader)
        logger.info(f"  Saved enhanced images for {image_filename}")
    
    return results, resolved_gamma, comparison_figure_path
