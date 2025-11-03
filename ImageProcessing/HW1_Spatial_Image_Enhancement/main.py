# Made By Chien-Hsun Chang (614410073) at 2025-11-01
# Course: Image Processing at CCU
# Assignment: Homework 1 - Spatial Image Enhancement

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

# Using Function Written by Myself (Not Using OpenCV or PIL Image Enhancement Functions)
# Please take a look at the source code in src/enhancement/ for details
from src.utils.logging_config import setup_logging, get_logger
from src.utils.image_utils import ImageFileLoader
from src.ui.visualization import ImageEnhancementVisualizer
from src.ui.gui import ImageReviewApp, ProcessedItem
from src.pipeline.processing_pipeline import (
    process_single_image,
    visualize_results,
    save_histogram_figures,
)


def main():
    """Run enhancement workflow for all test images and persist results."""
    setup_logging(level='INFO')
    logger = get_logger(__name__)
    
    try:
        gamma_value: Optional[float] = None  # None = enable adaptive gamma selection
        auto_gamma_bounds: Tuple[float, float] = (0.35, 1.8)
        target_mean_intensity = 0.6
        
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
        image_names = image_file_loader.list_available_images()
        if not image_names:
            raise FileNotFoundError(
                "No supported test images found. Add files with extensions bmp/png/jpg/jpeg/tif/tiff to `test_image/`."
            )

        app = ImageReviewApp(processed_items=None, gamma_value=gamma_value)

        class TkinterLogHandler(logging.Handler):
            def __init__(self, gui_app: ImageReviewApp):
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
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
        logging.getLogger().addHandler(gui_handler)

        processing_error = threading.Event()

        def processing_worker():
            try:
                app.schedule_processing_message("Loading test images...")
                logger.info("Loading test images...")
                enhancement_visualizer = ImageEnhancementVisualizer()
                loaded_images_dictionary = image_file_loader.load_multiple_image_files(image_names)

                processed_items: List[ProcessedItem] = []
                total_images = len(image_names)

                for index, image_filename in enumerate(image_names, start=1):
                    app.schedule_processing_message(f"Processing {index}/{total_images}: {image_filename}")
                    original_image = loaded_images_dictionary[image_filename]
                    results, resolved_gamma = process_single_image(
                        image_filename=image_filename,
                        image_array=original_image,
                        gamma_value=gamma_value,
                        logger=logger,
                        visualizer=enhancement_visualizer,
                        loader=image_file_loader,
                        visualize=False,
                        auto_gamma_bounds=auto_gamma_bounds,
                        target_mean_intensity=target_mean_intensity
                    )

                    figure_path = visualize_results(
                        image_filename=image_filename,
                        original_image=original_image,
                        results=results,
                        visualizer=enhancement_visualizer,
                        gamma_value=resolved_gamma,
                        display_plot_immediately=False
                    )
                    logger.info(f"Saved comparison figure to {figure_path}")
                    histogram_paths = save_histogram_figures(
                        image_filename=image_filename,
                        original_image=original_image,
                        results=results
                    )
                    if histogram_paths:
                        first_hist_path = next(iter(histogram_paths.values()))
                        logger.info("Saved histogram figures to %s", os.path.dirname(first_hist_path) or '.')
                    else:
                        logger.warning("No histogram figures generated for %s", image_filename)
                    processed_items.append(ProcessedItem(
                        filename=image_filename,
                        comparison_figure_path=figure_path,
                        gamma_value=resolved_gamma
                    ))
                    logger.info("Resolved gamma for %s: %.3f", image_filename, resolved_gamma)

                logger.info("=" * 60)
                logger.info("All image processing completed successfully!")
                logger.info("Results saved in 'results/' directory:")
                logger.info("- Processed images ready for interactive review")
                logger.info("- Histograms generated for all enhancement techniques")
                logger.info("- All processed images saved as .bmp files")
                logger.info("- Comparison figures saved as .png files")
                logger.info("=" * 60)

                app.schedule_processed_items(processed_items)
                app.schedule_processing_message("Processing complete")

            except Exception as worker_exception:  # pragma: no cover - background worker
                processing_error.set()
                logger.error(f"Error occurred during processing: {worker_exception}")
                logger.error("Program terminated with errors.")
                app.schedule_error(str(worker_exception))

        threading.Thread(target=processing_worker, daemon=True).start()

        app.run()

        logging.getLogger().removeHandler(gui_handler)
        gui_handler.close()

        if processing_error.is_set():
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error occurred during processing: {e}")
        logger.error("Program terminated with errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()