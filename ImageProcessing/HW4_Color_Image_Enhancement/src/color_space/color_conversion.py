"""
RGB and HSI Color Space Conversion Module

This module provides functions for converting between RGB and HSI color spaces.
All conversions are implemented manually without using OpenCV or PIL image processing functions.

HSI (Hue, Saturation, Intensity) color model:
- Hue (H): The color type, represented as an angle from 0 to 360 degrees
- Saturation (S): The purity of the color, from 0 to 1
- Intensity (I): The brightness of the color, from 0 to 1

References:
- Gonzalez & Woods, "Digital Image Processing", Chapter 6
"""

import numpy as np
import logging
import math

logger = logging.getLogger(__name__)


class RGBToHSIConverter:
    """Convert RGB color images to HSI color space."""

    def __init__(self):
        logger.debug("RGBToHSIConverter initialized")

    def convert(self, rgb_image):
        """
        Convert RGB image to HSI color space.
        
        Args:
            rgb_image: numpy array of shape (H, W, 3) with RGB values in [0, 255]
            
        Returns:
            hsi_image: numpy array of shape (H, W, 3) where:
                - H channel: Hue in [0, 360) degrees
                - S channel: Saturation in [0, 1]
                - I channel: Intensity in [0, 1]
        """
        logger.debug(f"Converting RGB image of shape {rgb_image.shape} to HSI")
        
        # Normalize RGB to [0, 1]
        rgb_normalized = rgb_image.astype(np.float64) / 255.0
        
        rows, cols, _ = rgb_normalized.shape
        hsi_image = np.zeros((rows, cols, 3), dtype=np.float64)
        
        for i in range(rows):
            for j in range(cols):
                r, g, b = rgb_normalized[i, j, 0], rgb_normalized[i, j, 1], rgb_normalized[i, j, 2]
                h, s, intensity = self._rgb_pixel_to_hsi(r, g, b)
                hsi_image[i, j, 0] = h
                hsi_image[i, j, 1] = s
                hsi_image[i, j, 2] = intensity
        
        logger.debug("RGB to HSI conversion completed")
        return hsi_image

    def _rgb_pixel_to_hsi(self, r, g, b):
        """Convert a single RGB pixel to HSI."""
        # Calculate Intensity
        intensity = (r + g + b) / 3.0
        
        # Calculate Saturation
        min_rgb = min(r, g, b)
        if intensity == 0:
            saturation = 0.0
        else:
            saturation = 1.0 - (min_rgb / intensity)
        
        # Calculate Hue
        if saturation == 0:
            hue = 0.0  # Undefined, set to 0
        else:
            numerator = 0.5 * ((r - g) + (r - b))
            denominator = math.sqrt((r - g) ** 2 + (r - b) * (g - b))
            
            if denominator == 0:
                hue = 0.0
            else:
                theta = math.acos(max(-1.0, min(1.0, numerator / denominator)))
                theta_degrees = math.degrees(theta)
                
                if b <= g:
                    hue = theta_degrees
                else:
                    hue = 360.0 - theta_degrees
        
        return hue, saturation, intensity


class HSIToRGBConverter:
    """Convert HSI color images back to RGB color space."""

    def __init__(self):
        logger.debug("HSIToRGBConverter initialized")

    def convert(self, hsi_image):
        """
        Convert HSI image to RGB color space.
        
        Args:
            hsi_image: numpy array of shape (H, W, 3) where:
                - H channel: Hue in [0, 360) degrees
                - S channel: Saturation in [0, 1]
                - I channel: Intensity in [0, 1]
                
        Returns:
            rgb_image: numpy array of shape (H, W, 3) with RGB values in [0, 255]
        """
        logger.debug(f"Converting HSI image of shape {hsi_image.shape} to RGB")
        
        rows, cols, _ = hsi_image.shape
        rgb_image = np.zeros((rows, cols, 3), dtype=np.float64)
        
        for i in range(rows):
            for j in range(cols):
                h, s, intensity = hsi_image[i, j, 0], hsi_image[i, j, 1], hsi_image[i, j, 2]
                r, g, b = self._hsi_pixel_to_rgb(h, s, intensity)
                rgb_image[i, j, 0] = r
                rgb_image[i, j, 1] = g
                rgb_image[i, j, 2] = b
        
        # Scale back to [0, 255] and clip
        rgb_image = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
        
        logger.debug("HSI to RGB conversion completed")
        return rgb_image

    def _hsi_pixel_to_rgb(self, h, s, intensity):
        """Convert a single HSI pixel to RGB."""
        # Normalize hue to [0, 360)
        h = h % 360.0
        
        if s == 0:
            # Achromatic case (grayscale)
            return intensity, intensity, intensity
        
        # Convert hue to radians for calculation
        h_rad = math.radians(h)
        
        if 0 <= h < 120:
            # RG sector
            b = intensity * (1 - s)
            r = intensity * (1 + (s * math.cos(h_rad)) / math.cos(math.radians(60) - h_rad))
            g = 3 * intensity - (r + b)
        elif 120 <= h < 240:
            # GB sector
            h_rad = math.radians(h - 120)
            r = intensity * (1 - s)
            g = intensity * (1 + (s * math.cos(h_rad)) / math.cos(math.radians(60) - h_rad))
            b = 3 * intensity - (r + g)
        else:
            # BR sector (240 <= h < 360)
            h_rad = math.radians(h - 240)
            g = intensity * (1 - s)
            b = intensity * (1 + (s * math.cos(h_rad)) / math.cos(math.radians(60) - h_rad))
            r = 3 * intensity - (g + b)
        
        # Clip values to valid range
        r = max(0.0, min(1.0, r))
        g = max(0.0, min(1.0, g))
        b = max(0.0, min(1.0, b))
        
        return r, g, b


def rgb_to_hsi(rgb_image):
    """Functional wrapper for RGB to HSI conversion."""
    converter = RGBToHSIConverter()
    return converter.convert(rgb_image)


def hsi_to_rgb(hsi_image):
    """Functional wrapper for HSI to RGB conversion."""
    converter = HSIToRGBConverter()
    return converter.convert(hsi_image)
