import numpy as np
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)


class ColorImageFileLoader:
    """Utility for loading and saving color images from a base directory."""

    SUPPORTED_EXTENSIONS = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')

    def __init__(self, base_directory_path='test_image'):
        """Store base directory path containing image assets."""
        self.base_directory_path = base_directory_path
        logger.info(f"ColorImageFileLoader initialized with base_directory_path='{base_directory_path}'")

    def list_available_images(self):
        """Return sorted list of available image filenames in the base directory."""
        if not os.path.isdir(self.base_directory_path):
            raise FileNotFoundError(f"Image directory not found: {self.base_directory_path}")

        filenames = []
        for entry in os.listdir(self.base_directory_path):
            complete_path = os.path.join(self.base_directory_path, entry)
            _, ext = os.path.splitext(entry)
            if os.path.isfile(complete_path) and ext.lower() in self.SUPPORTED_EXTENSIONS:
                filenames.append(entry)

        if not filenames:
            logger.warning(f"No supported image files found in {self.base_directory_path}")

        filenames.sort()
        logger.info(f"Discovered {len(filenames)} image file(s) for processing")
        return filenames

    def load_single_color_image(self, image_filename):
        """Return color image as float64 ndarray with shape (H, W, 3) in RGB format."""
        complete_image_path = os.path.join(self.base_directory_path, image_filename)
        if not os.path.exists(complete_image_path):
            raise FileNotFoundError(f"Image file not found: {complete_image_path}")
        
        loaded_image = Image.open(complete_image_path)
        # Convert to RGB if necessary
        if loaded_image.mode != 'RGB':
            loaded_image = loaded_image.convert('RGB')
        
        image_array = np.array(loaded_image, dtype=np.float64)
        logger.info(f"Loaded {image_filename}: {image_array.shape}")
        return image_array

    def load_multiple_color_images(self, image_filename_list):
        """Load list of color images and return dict keyed by filename."""
        loaded_images_dictionary = {}
        logger.info("Loading multiple color images...")
        for image_filename in image_filename_list:
            try:
                loaded_images_dictionary[image_filename] = self.load_single_color_image(image_filename)
            except Exception as e:
                logger.error(f"Failed to load {image_filename}: {e}")
                raise
        logger.info(f"Successfully loaded {len(loaded_images_dictionary)} images")
        return loaded_images_dictionary

    def save_color_image_array(self, image_array, output_filename, output_directory_path='results'):
        """Persist color ndarray as image file (auto-create directory)."""
        os.makedirs(output_directory_path, exist_ok=True)
        complete_output_path = os.path.join(output_directory_path, output_filename)
        
        # Ensure valid range and dtype
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image_array, mode='RGB')
        pil_image.save(complete_output_path)
        logger.debug(f"Saved image to {complete_output_path}")


class ColorHistogramCalculator:
    """Helpers to compute histogram counts for color images."""

    @staticmethod
    def calculate_channel_histogram(channel_array):
        """Return 256-bin frequency list for uint8 intensity values of a single channel."""
        rows, cols = channel_array.shape
        histogram = [0] * 256
        for i in range(rows):
            for j in range(cols):
                value = int(channel_array[i, j])
                if 0 <= value <= 255:
                    histogram[value] += 1
        return histogram

    @staticmethod
    def calculate_rgb_histograms(color_image):
        """Return dict with R, G, B channel histograms."""
        if color_image.dtype != np.uint8:
            color_image = np.clip(color_image, 0, 255).astype(np.uint8)
        
        r_channel = color_image[:, :, 0]
        g_channel = color_image[:, :, 1]
        b_channel = color_image[:, :, 2]
        
        return {
            'R': ColorHistogramCalculator.calculate_channel_histogram(r_channel),
            'G': ColorHistogramCalculator.calculate_channel_histogram(g_channel),
            'B': ColorHistogramCalculator.calculate_channel_histogram(b_channel)
        }

    @staticmethod
    def calculate_intensity_histogram(color_image):
        """Return histogram of grayscale intensity (average of R, G, B)."""
        if color_image.dtype != np.uint8:
            color_image = np.clip(color_image, 0, 255).astype(np.uint8)
        
        rows, cols, _ = color_image.shape
        histogram = [0] * 256
        for i in range(rows):
            for j in range(cols):
                # Calculate intensity as average of RGB
                r = int(color_image[i, j, 0])
                g = int(color_image[i, j, 1])
                b = int(color_image[i, j, 2])
                intensity = int((r + g + b) / 3)
                if 0 <= intensity <= 255:
                    histogram[intensity] += 1
        return histogram
