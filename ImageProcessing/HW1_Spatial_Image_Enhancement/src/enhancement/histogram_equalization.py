import numpy as np
import logging

logger = logging.getLogger(__name__)

class HistogramEqualizationProcessor:
    """Compute and apply manual histogram equalization to a grayscale image."""

    def __init__(self):
        """Initialize processor (no state kept)."""
        logger.debug("HistogramEqualizationProcessor initialized")

    def calculate_image_histogram_distribution(self, image):
        """Return 256-bin intensity histogram for a grayscale uint8 image."""
        rows, cols = image.shape
        histogram = [0] * 256
        for i in range(rows):
            for j in range(cols):
                pixel_value = int(image[i, j])
                histogram[pixel_value] += 1
        logger.debug("Histogram calculation completed")
        return histogram

    def calculate_cumulative_distribution_function(self, histogram_distribution):
        """Return cumulative distribution and its first non-zero value (cdf_min)."""
        cumulative_distribution_function = [0] * 256
        cumulative_distribution_function[0] = histogram_distribution[0]
        for k in range(1, 256):
            cumulative_distribution_function[k] = cumulative_distribution_function[k-1] + histogram_distribution[k]
        minimum_cdf_value = 0
        for k in range(256):
            if cumulative_distribution_function[k] > 0:
                minimum_cdf_value = cumulative_distribution_function[k]
                break
        logger.debug(f"CDF calculation completed, cdf_min={minimum_cdf_value}")
        return cumulative_distribution_function, minimum_cdf_value

    def apply_histogram_equalization(self, image):
        """Return histogram-equalized image computed via explicit loops."""
        rows, cols = image.shape
        total_pixels = rows * cols
        logger.debug(f"Equalizing histogram for {rows}x{cols} image")
        histogram_distribution = self.calculate_image_histogram_distribution(image)
        cumulative_distribution_function, minimum_cdf_value = self.calculate_cumulative_distribution_function(histogram_distribution)
        equalized_image_result = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                original_pixel_value = int(image[i, j])
                equalized_pixel_value = ((cumulative_distribution_function[original_pixel_value] - minimum_cdf_value) * 255) / (total_pixels - minimum_cdf_value)
                if equalized_pixel_value > 255:
                    equalized_pixel_value = 255
                elif equalized_pixel_value < 0:
                    equalized_pixel_value = 0
                equalized_image_result[i, j] = int(equalized_pixel_value)
        logger.debug("Histogram equalization completed")
        return equalized_image_result


def apply_histogram_equalization_enhancement(image):
    """Functional wrapper for one-off histogram equalization."""
    histogram_equalizer = HistogramEqualizationProcessor()
    return histogram_equalizer.apply_histogram_equalization(image)