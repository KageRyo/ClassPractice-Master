# Made By Chien-Hsun Chang (614410073) at 2025-09-29
# Course: Image Processing at CCU
# Assignment: Homework 1 - Spatial Image Enhancement

import sys
import os
import numpy as np

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
        image_file_loader = ImageFileLoader(base_directory_path='test_image')
        enhancement_visualizer = ImageEnhancementVisualizer()
        image_names = ['Cameraman.bmp', 'Jetplane.bmp', 'Lake.bmp', 'Peppers.bmp']
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