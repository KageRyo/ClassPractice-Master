# Made By Chien-Hsun Chang (614410073) at 2025-11-11
# Course: Image Processing at CCU
# Assignment: Homework 2 - Image Sharpening

import logging
import os
import sys
import threading
from typing import List, Optional


def get_resource_path(relative_path: str) -> str:
	"""Return absolute path for resources (supports PyInstaller bundles)."""
	base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
	return os.path.join(base_path, relative_path)


src_path = get_resource_path('src')
if os.path.exists(src_path):
	sys.path.insert(0, src_path)
else:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline.processing_pipeline import (  # noqa: E402
	SharpeningParameters,
	process_single_image,
	visualize_results,
)
from src.ui.gui import ImageReviewApp, ProcessedItem  # noqa: E402
from src.ui.visualization import ImageSharpeningVisualizer  # noqa: E402
from src.utils.image_utils import ImageFileLoader  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402


def resolve_test_image_path(logger: logging.Logger) -> str:
	candidate_paths = [
		'test_image',
		get_resource_path('test_image'),
		os.path.join('..', 'test_image'),
		os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_image'),
	]
	for path in candidate_paths:
		if os.path.exists(path) and os.path.isdir(path):
			logger.info('Found test images at: %s', path)
			return path
	raise FileNotFoundError('No test_image directory found in expected locations.')


def main():
	setup_logging(level='INFO')
	logger = get_logger(__name__)

	try:
		params = SharpeningParameters()
		parameter_summary = params.summarize()

		image_directory = resolve_test_image_path(logger)
		image_loader = ImageFileLoader(base_directory_path=image_directory)
		image_names = list(image_loader.list_available_images())
		if not image_names:
			raise FileNotFoundError(
				'No supported test images found. Add bmp/png/jpg/jpeg/tif/tiff files to test_image/. '
				'Execution aborted.'
			)

		app = ImageReviewApp(processed_items=None, parameter_summary=parameter_summary)

		class TkinterLogHandler(logging.Handler):  # pragma: no cover - GUI bridge
			def __init__(self, gui_app: ImageReviewApp):
				super().__init__()
				self.gui_app = gui_app

			def emit(self, record: logging.LogRecord) -> None:
				try:
					message = self.format(record)
					self.gui_app.schedule_log_message(message)
				except Exception:  # pragma: no cover - defensive
					self.handleError(record)

		gui_handler = TkinterLogHandler(app)
		gui_handler.setLevel(logging.INFO)
		gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'))
		logging.getLogger().addHandler(gui_handler)

		processing_error = threading.Event()

		def processing_worker():  # pragma: no cover - background thread
			try:
				app.schedule_processing_message('Loading test images...')
				logger.info('Loading test images...')
				visualizer = ImageSharpeningVisualizer()
				loaded_images = image_loader.load_multiple_image_files(image_names)

				processed_items: List[ProcessedItem] = []
				total_images = len(image_names)

				for index, image_filename in enumerate(image_names, start=1):
					app.schedule_processing_message(f'Processing {index}/{total_images}: {image_filename}')
					original_image = loaded_images[image_filename]
					results, stats = process_single_image(
						image_filename=image_filename,
						image_array=original_image,
						params=params,
						logger=logger,
						visualizer=visualizer,
						loader=image_loader,
						visualize=False,
						save=True,
					)

					figure_path = visualize_results(
						image_filename=image_filename,
						original_image=original_image,
						results=results,
						visualizer=visualizer,
						display_plot_immediately=False,
					)

					detail_text = (
						f"{parameter_summary} | "
						f"Means: orig={stats['original_mean']:.1f}, lap={stats['laplacian_mean']:.1f}, "
						f"unsharp={stats['unsharp_mean']:.1f}, high={stats['high_boost_mean']:.1f}, "
						f"homo={stats['homomorphic_mean']:.1f}"
					)

					processed_items.append(
						ProcessedItem(
							filename=image_filename,
							comparison_figure_path=figure_path,
							detail_text=detail_text,
						)
					)

				app.schedule_processed_items(processed_items)
				app.schedule_processing_message('Processing complete')

				logger.info('=' * 60)
				logger.info('All sharpening operations completed successfully!')
				logger.info('Results saved in results/ directory (BMP outputs and comparison figures).')
				logger.info('Default sharpening parameters: %s', parameter_summary)
				logger.info('=' * 60)

			except Exception as worker_exception:
				processing_error.set()
				logger.error('Error occurred during processing: %s', worker_exception)
				app.schedule_error(str(worker_exception))

		threading.Thread(target=processing_worker, daemon=True).start()

		app.run()

		logging.getLogger().removeHandler(gui_handler)
		gui_handler.close()

		if processing_error.is_set():
			sys.exit(1)

	except Exception as main_exception:
		logger.error('Program terminated with errors: %s', main_exception)
		sys.exit(1)


if __name__ == '__main__':
	main()
