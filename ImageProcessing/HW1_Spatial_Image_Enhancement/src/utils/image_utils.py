import numpy as np
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

class ImageFileLoader:
    """Utility for loading and saving grayscale images from a base directory."""

    def __init__(self, base_directory_path='test_image'):
        """Store base directory path containing image assets."""
        self.base_directory_path = base_directory_path
        logger.info(f"ImageFileLoader initialized with base_directory_path='{base_directory_path}'")

    def load_single_image_file(self, image_filename, convert_to_grayscale=True):
        """Return image as float64 ndarray; convert to grayscale if requested."""
        complete_image_path = os.path.join(self.base_directory_path, image_filename)
        if not os.path.exists(complete_image_path):
            raise FileNotFoundError(f"Image file not found: {complete_image_path}")
        loaded_image = Image.open(complete_image_path)
        if convert_to_grayscale:
            loaded_image = loaded_image.convert('L')
        image_array = np.array(loaded_image, dtype=np.float64)
        logger.info(f"Loaded {image_filename}: {image_array.shape}")
        return image_array

    def load_multiple_image_files(self, image_filename_list):
        """Load list of images and return dict keyed by filename."""
        loaded_images_dictionary = {}
        logger.info("Loading multiple images...")
        for image_filename in image_filename_list:
            try:
                loaded_images_dictionary[image_filename] = self.load_single_image_file(image_filename)
            except Exception as e:
                logger.error(f"Failed to load {image_filename}: {e}")
                raise
        logger.info(f"Successfully loaded {len(loaded_images_dictionary)} images")
        return loaded_images_dictionary

    def save_image_array_to_file(self, image_array, output_filename, output_directory_path='results'):
        """Persist ndarray as image file (auto-create directory)."""
        os.makedirs(output_directory_path, exist_ok=True)
        complete_output_path = os.path.join(output_directory_path, output_filename)
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        pil_image.save(complete_output_path)
        logger.debug(f"Saved image to {complete_output_path}")

class ImageHistogramCalculator:
    """Helpers to compute histogram counts and basic statistics."""

    @staticmethod
    def calculate_image_pixel_histogram(image_array):
        """Return 256-bin frequency list for uint8 intensity values."""
        image_rows, image_columns = image_array.shape
        pixel_intensity_histogram = [0] * 256
        for i in range(image_rows):
            for j in range(image_columns):
                pixel_intensity_value = int(image_array[i, j])
                if 0 <= pixel_intensity_value <= 255:
                    pixel_intensity_histogram[pixel_intensity_value] += 1
        return pixel_intensity_histogram

    @staticmethod
    def calculate_histogram_statistical_metrics(histogram_data):
        """Return dict with mean, std, min, max, total pixel count from histogram."""
        total_pixel_count = sum(histogram_data)
        weighted_intensity_sum = sum(i * count for i, count in enumerate(histogram_data))
        mean_intensity = weighted_intensity_sum / total_pixel_count if total_pixel_count > 0 else 0
        intensity_variance = sum(count * (i - mean_intensity) ** 2 for i, count in enumerate(histogram_data)) / total_pixel_count if total_pixel_count > 0 else 0
        standard_deviation = intensity_variance ** 0.5
        minimum_intensity_value = next((i for i, count in enumerate(histogram_data) if count > 0), 0)
        maximum_intensity_value = next((255 - i for i, count in enumerate(reversed(histogram_data)) if count > 0), 255)
        return {
            'mean': mean_intensity,
            'std': standard_deviation,
            'min': minimum_intensity_value,
            'max': maximum_intensity_value,
            'total_pixels': total_pixel_count
        }