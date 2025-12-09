import logging
import os
from typing import Dict, Iterable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageFileLoader:
    """Load and store grayscale images from a base directory."""

    SUPPORTED_EXTENSIONS = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')

    def __init__(self, base_directory_path: str = 'test_image'):
        self.base_directory_path = base_directory_path
        logger.info("ImageFileLoader initialized with base_directory_path='%s'", base_directory_path)

    def list_available_images(self) -> Iterable[str]:
        """Return sorted iterable of filenames located under the base path."""
        if not os.path.isdir(self.base_directory_path):
            raise FileNotFoundError(f"Image directory not found: {self.base_directory_path}")

        filenames = []
        for entry in os.listdir(self.base_directory_path):
            complete_path = os.path.join(self.base_directory_path, entry)
            _, ext = os.path.splitext(entry)
            if os.path.isfile(complete_path) and ext.lower() in self.SUPPORTED_EXTENSIONS:
                filenames.append(entry)

        if not filenames:
            logger.warning("No supported image files found in %s", self.base_directory_path)

        filenames.sort()
        logger.info("Discovered %d image file(s) for processing", len(filenames))
        return filenames

    def load_single_image_file(self, image_filename: str, convert_to_grayscale: bool = True) -> np.ndarray:
        """Return image as float64 ndarray; convert to grayscale if requested."""
        complete_image_path = os.path.join(self.base_directory_path, image_filename)
        if not os.path.exists(complete_image_path):
            raise FileNotFoundError(f"Image file not found: {complete_image_path}")
        loaded_image = Image.open(complete_image_path)
        if convert_to_grayscale:
            loaded_image = loaded_image.convert('L')
        image_array = np.array(loaded_image, dtype=np.float64)
        logger.info("Loaded %s: %s", image_filename, image_array.shape)
        return image_array

    def load_multiple_image_files(self, image_filename_list: Iterable[str]) -> Dict[str, np.ndarray]:
        """Load a collection of images and return dictionary keyed by filename."""
        loaded_images_dictionary: Dict[str, np.ndarray] = {}
        logger.info("Loading multiple images...")
        for image_filename in image_filename_list:
            loaded_images_dictionary[image_filename] = self.load_single_image_file(image_filename)
        logger.info("Successfully loaded %d images", len(loaded_images_dictionary))
        return loaded_images_dictionary

    def save_image_array_to_file(
        self,
        image_array: np.ndarray,
        output_filename: str,
        output_directory_path: str = 'results',
    ) -> str:
        """Persist ndarray as image file (auto-create directory)."""
        os.makedirs(output_directory_path, exist_ok=True)
        complete_output_path = os.path.join(output_directory_path, output_filename)
        if image_array.dtype != np.uint8:
            image_array = np.clip(np.rint(image_array), 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        pil_image.save(complete_output_path)
        logger.debug("Saved image to %s", complete_output_path)
        return complete_output_path
