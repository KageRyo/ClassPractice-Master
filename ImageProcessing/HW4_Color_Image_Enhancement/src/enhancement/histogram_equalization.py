"""
Histogram Equalization for Color Images

This module provides histogram equalization techniques for color images:
1. RGB Histogram Equalization: Apply histogram equalization to each RGB channel independently
2. HSI Intensity Histogram Equalization: Apply histogram equalization only to the Intensity channel
   in HSI color space (preserving Hue)

Note: All implementations are manual without using OpenCV or PIL image processing functions.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class RGBHistogramEqualization:
    """Apply histogram equalization to each RGB channel independently."""

    def __init__(self):
        logger.debug("RGBHistogramEqualization initialized")

    def equalize(self, rgb_image):
        """
        Apply histogram equalization to each RGB channel.
        
        Args:
            rgb_image: numpy array of shape (H, W, 3) with RGB values in [0, 255]
            
        Returns:
            equalized_image: numpy array of shape (H, W, 3) with equalized RGB values
        """
        logger.info("Applying RGB histogram equalization...")
        
        rows, cols, _ = rgb_image.shape
        equalized_image = np.zeros_like(rgb_image, dtype=np.uint8)
        
        for channel_idx in range(3):
            channel = rgb_image[:, :, channel_idx].astype(np.float64)
            equalized_channel = self._equalize_single_channel(channel, rows, cols)
            equalized_image[:, :, channel_idx] = equalized_channel
        
        logger.debug("RGB histogram equalization completed")
        return equalized_image

    def _equalize_single_channel(self, channel, rows, cols):
        """Apply histogram equalization to a single channel."""
        total_pixels = rows * cols
        
        # Calculate histogram
        histogram = [0] * 256
        for i in range(rows):
            for j in range(cols):
                value = int(channel[i, j])
                if 0 <= value <= 255:
                    histogram[value] += 1
        
        # Calculate CDF
        cdf = [0] * 256
        cdf[0] = histogram[0]
        for k in range(1, 256):
            cdf[k] = cdf[k-1] + histogram[k]
        
        # Find minimum non-zero CDF value
        cdf_min = 0
        for k in range(256):
            if cdf[k] > 0:
                cdf_min = cdf[k]
                break
        
        # Apply equalization
        equalized = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                original_value = int(channel[i, j])
                if total_pixels - cdf_min > 0:
                    new_value = ((cdf[original_value] - cdf_min) * 255) / (total_pixels - cdf_min)
                else:
                    new_value = original_value
                equalized[i, j] = int(max(0, min(255, new_value)))
        
        return equalized


class HSIIntensityHistogramEqualization:
    """Apply histogram equalization only to the Intensity channel in HSI space."""

    def __init__(self):
        logger.debug("HSIIntensityHistogramEqualization initialized")

    def equalize(self, hsi_image):
        """
        Apply histogram equalization to the Intensity (I) channel of HSI image.
        Hue and Saturation are preserved.
        
        Args:
            hsi_image: numpy array of shape (H, W, 3) where:
                - H channel: Hue in [0, 360)
                - S channel: Saturation in [0, 1]
                - I channel: Intensity in [0, 1]
                
        Returns:
            equalized_hsi: numpy array with equalized Intensity channel
        """
        logger.info("Applying HSI intensity histogram equalization...")
        
        rows, cols, _ = hsi_image.shape
        total_pixels = rows * cols
        
        # Copy HSI image
        equalized_hsi = hsi_image.copy()
        
        # Get Intensity channel and scale to [0, 255] for histogram calculation
        intensity_scaled = (hsi_image[:, :, 2] * 255).astype(np.float64)
        
        # Calculate histogram
        histogram = [0] * 256
        for i in range(rows):
            for j in range(cols):
                value = int(intensity_scaled[i, j])
                if 0 <= value <= 255:
                    histogram[value] += 1
        
        # Calculate CDF
        cdf = [0] * 256
        cdf[0] = histogram[0]
        for k in range(1, 256):
            cdf[k] = cdf[k-1] + histogram[k]
        
        # Find minimum non-zero CDF value
        cdf_min = 0
        for k in range(256):
            if cdf[k] > 0:
                cdf_min = cdf[k]
                break
        
        # Apply equalization to Intensity channel
        for i in range(rows):
            for j in range(cols):
                original_value = int(intensity_scaled[i, j])
                if total_pixels - cdf_min > 0:
                    new_value = ((cdf[original_value] - cdf_min) * 255) / (total_pixels - cdf_min)
                else:
                    new_value = original_value
                # Scale back to [0, 1]
                equalized_hsi[i, j, 2] = max(0.0, min(1.0, new_value / 255.0))
        
        logger.debug("HSI intensity histogram equalization completed")
        return equalized_hsi


def apply_rgb_histogram_equalization(rgb_image):
    """Functional wrapper for RGB histogram equalization."""
    equalizer = RGBHistogramEqualization()
    return equalizer.equalize(rgb_image)


def apply_hsi_intensity_histogram_equalization(hsi_image):
    """Functional wrapper for HSI intensity histogram equalization."""
    equalizer = HSIIntensityHistogramEqualization()
    return equalizer.equalize(hsi_image)
