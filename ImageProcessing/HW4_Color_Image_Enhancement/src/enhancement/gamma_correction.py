"""
Gamma Correction for Color Images

This module provides gamma correction techniques for color images:
1. RGB Gamma Correction: Apply gamma correction to each RGB channel independently
2. HSI Intensity Gamma Correction: Apply gamma correction only to the Intensity channel
   in HSI color space (preserving Hue)

Note: All implementations are manual without using OpenCV or PIL image processing functions.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class RGBGammaCorrection:
    """Apply gamma correction to each RGB channel independently."""

    def __init__(self, gamma=1.0):
        """
        Initialize gamma correction processor.
        
        Args:
            gamma: Gamma value for power-law transformation.
                   gamma < 1: brightens the image
                   gamma > 1: darkens the image
        """
        self.gamma = gamma
        logger.debug(f"RGBGammaCorrection initialized with gamma={gamma}")

    def apply(self, rgb_image):
        """
        Apply gamma correction to each RGB channel.
        
        Args:
            rgb_image: numpy array of shape (H, W, 3) with RGB values in [0, 255]
            
        Returns:
            corrected_image: numpy array of shape (H, W, 3) with gamma-corrected RGB values
        """
        logger.info(f"Applying RGB gamma correction with gamma={self.gamma}...")
        
        rows, cols, _ = rgb_image.shape
        corrected_image = np.zeros_like(rgb_image, dtype=np.uint8)
        
        for i in range(rows):
            for j in range(cols):
                for c in range(3):
                    # Normalize to [0, 1], apply gamma, scale back to [0, 255]
                    normalized = rgb_image[i, j, c] / 255.0
                    corrected = normalized ** self.gamma
                    corrected_image[i, j, c] = int(max(0, min(255, corrected * 255)))
        
        logger.debug("RGB gamma correction completed")
        return corrected_image


class HSIIntensityGammaCorrection:
    """Apply gamma correction only to the Intensity channel in HSI space."""

    def __init__(self, gamma=1.0):
        """
        Initialize gamma correction processor.
        
        Args:
            gamma: Gamma value for power-law transformation on Intensity channel.
        """
        self.gamma = gamma
        logger.debug(f"HSIIntensityGammaCorrection initialized with gamma={gamma}")

    def apply(self, hsi_image):
        """
        Apply gamma correction to the Intensity (I) channel of HSI image.
        Hue and Saturation are preserved.
        
        Args:
            hsi_image: numpy array of shape (H, W, 3) where:
                - H channel: Hue in [0, 360)
                - S channel: Saturation in [0, 1]
                - I channel: Intensity in [0, 1]
                
        Returns:
            corrected_hsi: numpy array with gamma-corrected Intensity channel
        """
        logger.info(f"Applying HSI intensity gamma correction with gamma={self.gamma}...")
        
        rows, cols, _ = hsi_image.shape
        corrected_hsi = hsi_image.copy()
        
        for i in range(rows):
            for j in range(cols):
                intensity = hsi_image[i, j, 2]
                corrected_intensity = intensity ** self.gamma
                corrected_hsi[i, j, 2] = max(0.0, min(1.0, corrected_intensity))
        
        logger.debug("HSI intensity gamma correction completed")
        return corrected_hsi


def apply_rgb_gamma_correction(rgb_image, gamma=1.0):
    """Functional wrapper for RGB gamma correction."""
    corrector = RGBGammaCorrection(gamma)
    return corrector.apply(rgb_image)


def apply_hsi_intensity_gamma_correction(hsi_image, gamma=1.0):
    """Functional wrapper for HSI intensity gamma correction."""
    corrector = HSIIntensityGammaCorrection(gamma)
    return corrector.apply(hsi_image)


def estimate_gamma_for_color_brightness(rgb_image, target_mean=0.5, min_gamma=0.3, max_gamma=2.5):
    """
    Estimate optimal gamma value to achieve target mean intensity for color image.
    
    Args:
        rgb_image: Input RGB image array
        target_mean: Target mean intensity (0-1 scale)
        min_gamma: Minimum gamma bound
        max_gamma: Maximum gamma bound
        
    Returns:
        Estimated gamma value
    """
    rows, cols, _ = rgb_image.shape
    
    # Calculate current mean intensity
    total_intensity = 0.0
    for i in range(rows):
        for j in range(cols):
            r, g, b = rgb_image[i, j, 0], rgb_image[i, j, 1], rgb_image[i, j, 2]
            total_intensity += (r + g + b) / 3.0
    
    current_mean = (total_intensity / (rows * cols)) / 255.0
    
    if current_mean <= 0 or current_mean >= 1:
        return 1.0
    
    # Estimate gamma: target_mean = current_mean^gamma
    # gamma = log(target_mean) / log(current_mean)
    import math
    try:
        gamma = math.log(target_mean) / math.log(current_mean)
        gamma = max(min_gamma, min(max_gamma, gamma))
    except (ValueError, ZeroDivisionError):
        gamma = 1.0
    
    logger.info(f"Estimated gamma={gamma:.3f} for target mean={target_mean}")
    return gamma
