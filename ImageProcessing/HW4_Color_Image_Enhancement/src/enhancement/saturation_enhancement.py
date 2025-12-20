"""
Saturation Enhancement for Color Images

This module provides saturation enhancement techniques in HSI color space.
The Hue component is preserved while Saturation is enhanced.

Note: All implementations are manual without using OpenCV or PIL image processing functions.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class SaturationEnhancement:
    """Enhance the Saturation component in HSI color space."""

    def __init__(self, factor=1.5):
        """
        Initialize saturation enhancement processor.
        
        Args:
            factor: Saturation multiplier.
                   factor > 1: increases saturation (more vivid colors)
                   factor < 1: decreases saturation (more muted colors)
        """
        self.factor = factor
        logger.debug(f"SaturationEnhancement initialized with factor={factor}")

    def enhance(self, hsi_image):
        """
        Enhance the Saturation (S) channel of HSI image.
        Hue and Intensity are preserved.
        
        Args:
            hsi_image: numpy array of shape (H, W, 3) where:
                - H channel: Hue in [0, 360)
                - S channel: Saturation in [0, 1]
                - I channel: Intensity in [0, 1]
                
        Returns:
            enhanced_hsi: numpy array with enhanced Saturation channel
        """
        logger.info(f"Applying saturation enhancement with factor={self.factor}...")
        
        rows, cols, _ = hsi_image.shape
        enhanced_hsi = hsi_image.copy()
        
        for i in range(rows):
            for j in range(cols):
                saturation = hsi_image[i, j, 1]
                enhanced_saturation = saturation * self.factor
                enhanced_hsi[i, j, 1] = max(0.0, min(1.0, enhanced_saturation))
        
        logger.debug("Saturation enhancement completed")
        return enhanced_hsi


def apply_saturation_enhancement(hsi_image, factor=1.5):
    """Functional wrapper for saturation enhancement."""
    enhancer = SaturationEnhancement(factor)
    return enhancer.enhance(hsi_image)
