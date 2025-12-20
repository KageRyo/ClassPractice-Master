# Made By Chien-Hsun Chang (614410073) at 2025-12-21
# Course: Image Processing at CCU
# Assignment: Homework 4 - Color Image Enhancement
#
# Note: This implementation uses manual algorithms for color image enhancement.
# No OpenCV or PIL image processing functions are used for the enhancement operations.
# RGB and HSI color space conversions are implemented from scratch.
# The Hue (H) component is preserved in all HSI-based enhancements.

import sys
import os
import threading
import logging
from typing import List, Optional, Tuple

# Add resource path handling for PyInstaller
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

# Import custom modules
from src.utils.logging_config import setup_logging, get_logger
from src.utils.image_utils import ColorImageFileLoader
from src.ui.visualization import ColorEnhancementVisualizer
from src.ui.gui import ColorImageReviewApp, ProcessedItem
from src.pipeline.processing_pipeline import (
    process_single_color_image,
    visualize_color_results,
)


def main():
    """Run color enhancement workflow for all test images and persist results."""
    setup_logging(level='INFO')
    logger = get_logger(__name__)
    
    try:
        gamma_value: Optional[float] = None  # None = enable adaptive gamma selection
        auto_gamma_bounds: Tuple[float, float] = (0.3, 2.5)
        target_mean_intensity = 0.5
        
        # Try to find test_image directory in different possible locations
        test_image_path = 'test_image'
        possible_paths = [
            'test_image',
            os.path.join(os.path.dirname(__file__), 'test_image'),
            os.path.join(get_resource_path(''), 'test_image'),
        ]
        
        for path in possible_paths:
            if os.path.isdir(path):
                test_image_path = path
                break
        
        logger.info(f"Using test image directory: {test_image_path}")
        
        # Initialize components
        loader = ColorImageFileLoader(base_directory_path=test_image_path)
        visualizer = ColorEnhancementVisualizer(figure_size_dimensions=(20, 10), image_resolution_dpi=200)
        
        # Get list of available images
        image_files = loader.list_available_images()
        
        if not image_files:
            logger.error("No images found in test_image directory!")
            return
        
        logger.info(f"Found {len(image_files)} color image(s) to process")
        
        # Create results directory
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Process images and collect results
        processed_items: List[ProcessedItem] = []
        
        def process_images():
            """Background thread for processing images."""
            nonlocal processed_items
            
            for idx, filename in enumerate(image_files):
                try:
                    logger.info(f"[{idx+1}/{len(image_files)}] Processing {filename}...")
                    
                    # Load color image
                    image_array = loader.load_single_color_image(filename)
                    
                    # Process the image
                    results, resolved_gamma, comparison_path = process_single_color_image(
                        image_filename=filename,
                        image_array=image_array,
                        gamma_value=gamma_value,
                        logger=logger,
                        visualizer=visualizer,
                        loader=loader,
                        visualize=True,
                        save=True,
                        auto_gamma_bounds=auto_gamma_bounds,
                        target_mean_intensity=target_mean_intensity
                    )
                    
                    # Create processed item for GUI
                    technique_desc = (
                        "1. RGB Histogram Equalization (each channel independently)\n"
                        "2. HSI Intensity Histogram Equalization (preserves Hue)\n"
                        "3. HSI Intensity Gamma Correction (preserves Hue)\n"
                        "4. HSI Saturation Enhancement (preserves Hue)"
                    )
                    
                    processed_item = ProcessedItem(
                        filename=filename,
                        comparison_figure_path=comparison_path,
                        gamma_value=resolved_gamma,
                        technique_description=technique_desc
                    )
                    processed_items.append(processed_item)
                    
                    # Update GUI if running
                    if hasattr(app, 'update_processed_items'):
                        app.root.after(0, lambda items=processed_items.copy(): 
                                       app.update_processed_items(items))
                        app.root.after(0, lambda msg=f"Processed {filename}": 
                                       app.append_log(msg))
                    
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {e}")
                    if hasattr(app, 'append_log'):
                        app.root.after(0, lambda err=str(e), fn=filename: 
                                       app.append_log(f"ERROR: {fn} - {err}"))
            
            # Final update
            if hasattr(app, 'set_processing_message'):
                app.root.after(0, lambda: app.set_processing_message("All images processed!"))
                app.root.after(0, lambda: app.append_log(f"Completed processing {len(processed_items)} images"))
        
        # Create GUI
        app = ColorImageReviewApp(processed_items=None, gamma_value=gamma_value)
        
        # Start processing in background thread
        processing_thread = threading.Thread(target=process_images, daemon=True)
        processing_thread.start()
        
        # Run GUI main loop
        app.run()
        
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        raise


def run_without_gui():
    """Run color enhancement without GUI (for testing or batch processing)."""
    setup_logging(level='INFO')
    logger = get_logger(__name__)
    
    try:
        gamma_value: Optional[float] = None
        auto_gamma_bounds: Tuple[float, float] = (0.3, 2.5)
        target_mean_intensity = 0.5
        
        test_image_path = 'test_image'
        if not os.path.isdir(test_image_path):
            test_image_path = os.path.join(os.path.dirname(__file__), 'test_image')
        
        loader = ColorImageFileLoader(base_directory_path=test_image_path)
        visualizer = ColorEnhancementVisualizer(figure_size_dimensions=(20, 10), image_resolution_dpi=200)
        
        image_files = loader.list_available_images()
        
        if not image_files:
            logger.error("No images found!")
            return
        
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        for idx, filename in enumerate(image_files):
            logger.info(f"[{idx+1}/{len(image_files)}] Processing {filename}...")
            
            image_array = loader.load_single_color_image(filename)
            
            results, resolved_gamma, comparison_path = process_single_color_image(
                image_filename=filename,
                image_array=image_array,
                gamma_value=gamma_value,
                logger=logger,
                visualizer=visualizer,
                loader=loader,
                visualize=True,
                save=True,
                auto_gamma_bounds=auto_gamma_bounds,
                target_mean_intensity=target_mean_intensity
            )
            
            logger.info(f"  Completed with gamma={resolved_gamma:.3f}")
        
        logger.info(f"All {len(image_files)} images processed successfully!")
        
    except Exception as e:
        logger.exception(f"Error: {e}")
        raise


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--no-gui":
        run_without_gui()
    else:
        main()
