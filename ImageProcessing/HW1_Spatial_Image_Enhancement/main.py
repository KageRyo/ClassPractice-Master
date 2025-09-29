# Made By Chien-Hsun Chang (614410073) at 2025-09-29
# Course: Image Processing at CCU
# Assignment: Homework 1 - Spatial Image Enhancement

import sys
import os
import numpy as np

# Add better resource path handling for PyInstaller
def get_resource_path(relative_path):
    """Get the absolute path to a resource, works for dev and for PyInstaller"""
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# Add src to path
src_path = get_resource_path('src')
if os.path.exists(src_path):
    sys.path.insert(0, src_path)
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Using Function Written by Myself (Not Using OpenCV or PIL Image Enhancement Functions)
# Please take a look at the source code in src/enhancement/ for details
from src.utils.logging_config import setup_logging, get_logger
from src.utils.image_utils import ImageFileLoader
from src.ui.visualization import ImageEnhancementVisualizer
from src.pipeline.processing_pipeline import process_single_image


def main():
    """Run enhancement workflow for all test images and persist results."""
    setup_logging(level='INFO')
    logger = get_logger(__name__)
    
    try:
        gamma_value = 2.2  # Centralized gamma setting
        
        # Try to find test_image directory in different possible locations
        test_image_path = 'test_image'
        possible_paths = [
            'test_image',                           # Current directory
            get_resource_path('test_image'),        # Resource path
            os.path.join('..', 'test_image'),       # One level up
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_image')  # Same dir as script
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                test_image_path = path
                logger.info(f"Found test images at: {path}")
                break
        
        image_file_loader = ImageFileLoader(base_directory_path=test_image_path)
        enhancement_visualizer = ImageEnhancementVisualizer()
        image_names = ['Cameraman.bmp', 'Jetplane.bmp', 'Lake.bmp', 'Peppers.bmp']  # Download from eCourse2 at 2025-09-28
        logger.info("Loading test images...")
        loaded_images_dictionary = image_file_loader.load_multiple_image_files(image_names)

        for image_filename in image_names:
            process_single_image(
                image_filename=image_filename,
                image_array=loaded_images_dictionary[image_filename],
                gamma_value=gamma_value,
                logger=logger,
                visualizer=enhancement_visualizer,
                loader=image_file_loader
            )

        logger.info("=" * 60)
        logger.info("All image processing completed successfully!")
        logger.info("Results saved in 'results/' directory:")
        logger.info("- Original and processed images displayed")
        logger.info("- Histograms shown for all enhancement techniques")
        logger.info("- All processed images saved as .bmp files")
        logger.info("- Comparison figures saved as .png files")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error occurred during processing: {e}")
        logger.error("Program terminated with errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()