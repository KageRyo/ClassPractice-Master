"""Power-law (gamma) intensity transformation for image enhancement.

This module implements power-law transformation using the formula:
    s = c * (r/L)^gamma
where:
    - s is the output intensity
    - r is the input intensity
    - c is the scaling constant (usually 1.0)
    - L is the maximum intensity value (255 for uint8)
    - gamma controls the transformation curve

Gamma values:
    - gamma < 1: brightens the image (expands dark regions)
    - gamma = 1: no change (linear transformation)
    - gamma > 1: darkens the image (compresses dark regions)

Author: Chien-Hsun Chang (614410073)
Course: Image Processing at CCU
Assignment: Homework 1 - Spatial Image Enhancement
"""
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

class PowerLawTransformer:
    """Apply power-law (gamma) intensity transformation to grayscale images.
    
    The power-law transformation is defined as:
        output = c * (input/255)^gamma * 255
    
    This transformation is useful for:
    - Gamma correction for display devices
    - Enhancing dark or bright regions
    - Contrast adjustment
    
    Attributes:
        gamma_value (float): Gamma parameter controlling the transformation curve
        scaling_constant (float): Scaling factor (typically 1.0)
    """

    def __init__(self, gamma_value=2.2, scaling_constant=1.0):
        """Initialize the power-law transformer.
        
        Args:
            gamma_value (float, optional): Gamma parameter. Defaults to 2.2.
                - Values < 1.0 brighten the image
                - Values > 1.0 darken the image
            scaling_constant (float, optional): Scaling factor. Defaults to 1.0.
        """
        self.gamma_value = gamma_value
        self.scaling_constant = scaling_constant
        logger.info(f"PowerLawTransformer initialized with gamma_value={gamma_value}, scaling_constant={scaling_constant}")

    def transform(self, image):
        """Apply power-law transformation to input image.
        
        Args:
            image (np.ndarray): Input grayscale image as uint8 2D array
            
        Returns:
            np.ndarray: Transformed image as uint8 2D array with same shape as input
            
        Note:
            Uses explicit loops for educational purposes rather than vectorized operations.
            Values are clamped to [0, 255] range to prevent overflow.
        """
        rows, cols = image.shape
        result = np.zeros((rows, cols), dtype=np.uint8)
        logger.debug(f"Applying power-law transformation to {rows}x{cols} image")
        for i in range(rows):
            for j in range(cols):
                normalized_pixel_value = image[i, j] / 255.0
                transformed_pixel_value = self.scaling_constant * (normalized_pixel_value ** self.gamma_value)
                result_pixel_value = transformed_pixel_value * 255.0
                if result_pixel_value > 255:
                    result_pixel_value = 255
                elif result_pixel_value < 0:
                    result_pixel_value = 0
                result[i, j] = int(result_pixel_value)
        logger.debug("Power-law transformation completed")
        return result

def apply_power_law_transformation(image, gamma_value=2.2, scaling_constant=1.0):
    """Apply power-law transformation to an image (functional interface).
    
    Convenience function that creates a PowerLawTransformer instance and applies
    the transformation in a single call. Useful for one-off transformations.
    
    Args:
        image (np.ndarray): Input grayscale image as uint8 2D array
        gamma_value (float, optional): Gamma parameter. Defaults to 2.2.
        scaling_constant (float, optional): Scaling factor. Defaults to 1.0.
        
    Returns:
        np.ndarray: Transformed image as uint8 2D array
        
    Example:
        >>> import numpy as np
        >>> img = np.array([[100, 150, 200]], dtype=np.uint8)
        >>> enhanced = apply_power_law_transformation(img, gamma_value=0.5)  # Brighten
    """
    transformer = PowerLawTransformer(gamma_value, scaling_constant)
    return transformer.transform(image)


def estimate_gamma_for_brightness(
    image,
    target_mean=0.6,
    min_gamma=0.35,
    max_gamma=2.5,
):
    """Estimate a gamma value that nudges the image mean toward ``target_mean``.

    Args:
        image (np.ndarray): Input grayscale image (0-255 range expected).
        target_mean (float): Desired mean intensity after normalization (0-1).
        min_gamma (float): Lower bound for the returned gamma value.
        max_gamma (float): Upper bound for the returned gamma value.

    Returns:
        float: Gamma value clamped to ``[min_gamma, max_gamma]``.

    Notes:
        Uses a coarse approximation based on normalized mean intensity. The
        returned gamma brightens dark images (gamma < 1) while keeping overly
        aggressive corrections in check via the provided bounds.
    """
    normalized_image = np.asarray(image, dtype=np.float64) / 255.0
    normalized_image = np.clip(normalized_image, 1e-6, 1.0)

    mean_intensity = float(normalized_image.mean())
    if not math.isfinite(mean_intensity) or mean_intensity <= 1e-6:
        return float(min_gamma)

    sanitized_target = float(np.clip(target_mean, 1e-3, 0.95))
    mean_intensity = float(np.clip(mean_intensity, 1e-3, 0.95))

    try:
        estimated_gamma = math.log(sanitized_target) / math.log(mean_intensity)
    except ValueError:
        estimated_gamma = min_gamma

    bounded_gamma = float(np.clip(estimated_gamma, min_gamma, max_gamma))
    return bounded_gamma