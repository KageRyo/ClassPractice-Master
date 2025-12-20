"""
Intensity Enhancement for Color Images

This module provides intensity enhancement techniques in HSI color space.
Techniques include contrast stretching applied to the Intensity component.

Note: All implementations are manual without using OpenCV or PIL image processing functions.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class IntensityContrastStretching:
    """Apply contrast stretching to the Intensity component in HSI color space."""

    def __init__(self, low_percentile=1, high_percentile=99):
        """
        Initialize contrast stretching processor.
        
        Args:
            low_percentile: Lower percentile for contrast stretching (default 1%)
            high_percentile: Upper percentile for contrast stretching (default 99%)
        """
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        logger.debug(f"IntensityContrastStretching initialized with percentiles={low_percentile}-{high_percentile}")

    def stretch(self, hsi_image):
        """
        Apply contrast stretching to the Intensity (I) channel of HSI image.
        Hue and Saturation are preserved.
        
        Args:
            hsi_image: numpy array of shape (H, W, 3) where:
                - H channel: Hue in [0, 360)
                - S channel: Saturation in [0, 1]
                - I channel: Intensity in [0, 1]
                
        Returns:
            stretched_hsi: numpy array with contrast-stretched Intensity channel
        """
        logger.info("Applying intensity contrast stretching...")
        
        rows, cols, _ = hsi_image.shape
        stretched_hsi = hsi_image.copy()
        
        # Collect all intensity values
        intensities = []
        for i in range(rows):
            for j in range(cols):
                intensities.append(hsi_image[i, j, 2])
        
        # Sort to find percentiles manually
        intensities_sorted = sorted(intensities)
        total_pixels = len(intensities_sorted)
        
        low_idx = int(total_pixels * self.low_percentile / 100)
        high_idx = int(total_pixels * self.high_percentile / 100) - 1
        high_idx = max(low_idx + 1, high_idx)
        
        i_min = intensities_sorted[low_idx]
        i_max = intensities_sorted[min(high_idx, total_pixels - 1)]
        
        # Avoid division by zero
        if i_max - i_min < 1e-6:
            logger.warning("Intensity range too small for contrast stretching")
            return stretched_hsi
        
        # Apply contrast stretching to Intensity channel
        for i in range(rows):
            for j in range(cols):
                intensity = hsi_image[i, j, 2]
                # Linear stretch
                stretched_intensity = (intensity - i_min) / (i_max - i_min)
                stretched_hsi[i, j, 2] = max(0.0, min(1.0, stretched_intensity))
        
        logger.debug("Intensity contrast stretching completed")
        return stretched_hsi


class RGBContrastStretching:
    """Apply contrast stretching to each RGB channel independently."""

    def __init__(self, low_percentile=1, high_percentile=99):
        """
        Initialize contrast stretching processor.
        
        Args:
            low_percentile: Lower percentile for contrast stretching
            high_percentile: Upper percentile for contrast stretching
        """
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        logger.debug(f"RGBContrastStretching initialized with percentiles={low_percentile}-{high_percentile}")

    def stretch(self, rgb_image):
        """
        Apply contrast stretching to each RGB channel.
        
        Args:
            rgb_image: numpy array of shape (H, W, 3) with RGB values in [0, 255]
            
        Returns:
            stretched_image: numpy array with contrast-stretched RGB values
        """
        logger.info("Applying RGB contrast stretching...")
        
        rows, cols, _ = rgb_image.shape
        stretched_image = np.zeros_like(rgb_image, dtype=np.uint8)
        
        for c in range(3):
            channel = rgb_image[:, :, c].astype(np.float64)
            stretched_channel = self._stretch_single_channel(channel, rows, cols)
            stretched_image[:, :, c] = stretched_channel
        
        logger.debug("RGB contrast stretching completed")
        return stretched_image

    def _stretch_single_channel(self, channel, rows, cols):
        """Apply contrast stretching to a single channel."""
        # Collect all values
        values = []
        for i in range(rows):
            for j in range(cols):
                values.append(channel[i, j])
        
        # Sort to find percentiles
        values_sorted = sorted(values)
        total_pixels = len(values_sorted)
        
        low_idx = int(total_pixels * self.low_percentile / 100)
        high_idx = int(total_pixels * self.high_percentile / 100) - 1
        high_idx = max(low_idx + 1, high_idx)
        
        v_min = values_sorted[low_idx]
        v_max = values_sorted[min(high_idx, total_pixels - 1)]
        
        # Apply stretching
        stretched = np.zeros((rows, cols), dtype=np.uint8)
        if v_max - v_min < 1e-6:
            return channel.astype(np.uint8)
        
        for i in range(rows):
            for j in range(cols):
                value = channel[i, j]
                stretched_value = ((value - v_min) / (v_max - v_min)) * 255
                stretched[i, j] = int(max(0, min(255, stretched_value)))
        
        return stretched


def apply_intensity_contrast_stretching(hsi_image, low_percentile=1, high_percentile=99):
    """Functional wrapper for intensity contrast stretching."""
    stretcher = IntensityContrastStretching(low_percentile, high_percentile)
    return stretcher.stretch(hsi_image)


def apply_rgb_contrast_stretching(rgb_image, low_percentile=1, high_percentile=99):
    """Functional wrapper for RGB contrast stretching."""
    stretcher = RGBContrastStretching(low_percentile, high_percentile)
    return stretcher.stretch(rgb_image)
