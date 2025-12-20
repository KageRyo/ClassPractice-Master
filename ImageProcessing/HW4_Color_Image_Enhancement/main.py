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
            get_resource_path('test_image'),
        ]
        
        for path in possible_paths:
            if os.path.isdir(path):
                test_image_path = path
                break
        
        logger.info(f"Using test image directory: {test_image_path}")
        
        # Initialize image loader and check for images
        loader = ColorImageFileLoader(base_directory_path=test_image_path)
        image_files = loader.list_available_images()
        
        if not image_files:
            raise FileNotFoundError(
                "No supported test images found. Add PNG/JPG/BMP files to `test_image/` folder."
            )
        
        logger.info(f"Found {len(image_files)} color image(s) to process")
        
        # Create results directory
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Create GUI first (will show "Processing..." state)
        app = ColorImageReviewApp(processed_items=None, gamma_value=gamma_value)
        
        # Setup log handler to redirect logs to GUI
        class TkinterLogHandler(logging.Handler):
            def __init__(self, gui_app: ColorImageReviewApp):
                super().__init__()
                self.gui_app = gui_app

            def emit(self, record: logging.LogRecord):
                try:
                    message = self.format(record)
                    self.gui_app.schedule_log_message(message)
                except Exception:
                    self.handleError(record)

        gui_handler = TkinterLogHandler(app)
        gui_handler.setLevel(logging.INFO)
        gui_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'
        ))
        logging.getLogger().addHandler(gui_handler)
        
        # Error flag for background thread
        processing_error = threading.Event()
        
        def build_status_message(done_list, processing_name, wait_list):
            """Build status message showing done/processing/wait images."""
            parts = []
            if done_list:
                done_str = ", ".join(done_list)
                parts.append(f"Done: {done_str}")
            if processing_name:
                parts.append(f"Processing: {processing_name}")
            if wait_list:
                wait_str = ", ".join(wait_list)
                parts.append(f"Wait: {wait_str}")
            return " | ".join(parts)
        
        def processing_worker():
            """Background thread for processing all images."""
            try:
                status_msg = build_status_message([], None, image_files)
                app.schedule_processing_message(status_msg)
                logger.info("Loading test images...")
                
                visualizer = ColorEnhancementVisualizer(
                    figure_size_dimensions=(20, 10), 
                    image_resolution_dpi=200
                )
                
                processed_items: List[ProcessedItem] = []
                total_images = len(image_files)
                done_files: List[str] = []
                
                for idx, filename in enumerate(image_files):
                    # Done / Processing / Wait
                    wait_files = image_files[idx + 1:]
                    status_msg = build_status_message(done_files, filename, wait_files)
                    app.schedule_processing_message(status_msg)
                    logger.info(f"[{idx+1}/{total_images}] Processing {filename}...")
                    
                    # Load and process the image
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
                    
                    # Create processed item for GUI
                    technique_desc = (
                        "RGB Hist.Eq. | HSI Intensity Hist.Eq. | "
                        "HSI Gamma | HSI Saturation Enh."
                    )
                    
                    processed_item = ProcessedItem(
                        filename=filename,
                        comparison_figure_path=comparison_path,
                        gamma_value=resolved_gamma,
                        technique_description=technique_desc
                    )
                    processed_items.append(processed_item)
                    done_files.append(filename)
                    logger.info(f"  Completed {filename} with gamma={resolved_gamma:.3f}")
                    
                    app.schedule_processed_items(list(processed_items))
                
                # All processing complete
                logger.info("=" * 50)
                logger.info("All image processing completed successfully!")
                logger.info(f"Results saved in '{results_dir}/' directory")
                logger.info("=" * 50)
                
                app.schedule_processing_message(f"All {total_images} Images Complete")
                
            except Exception as e:
                processing_error.set()
                logger.error(f"Error during processing: {e}")
                app.schedule_error(str(e))
        
        # Start processing in background thread
        processing_thread = threading.Thread(target=processing_worker, daemon=True)
        processing_thread.start()
        
        # Run GUI main loop (blocks until window is closed)
        app.run()
        
        # Cleanup
        logging.getLogger().removeHandler(gui_handler)
        gui_handler.close()
        
        if processing_error.is_set():
            sys.exit(1)
        
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


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
