# Made By Chien-Hsun Chang (614410073) at 2025-12-10
# Course: Image Processing at CCU
# Assignment: Homework 3 - Image Restoration

import logging
import os
import sys
import threading
from typing import List, Optional, Tuple


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
    RestorationParameters,
    compute_restoration_outputs,
    visualize_results,
    save_results,
    collect_intensity_statistics,
    compute_psnr,
)
from src.ui.gui import ImageReviewApp, ProcessedItem  # noqa: E402
from src.ui.visualization import ImageRestorationVisualizer  # noqa: E402
from src.utils.image_utils import ImageFileLoader  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402


def resolve_test_image_path(logger: logging.Logger) -> str:
    """尋找測試影像目錄。"""
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


def pair_original_and_degraded_images(
    image_names: List[str],
    logger: logging.Logger,
) -> List[Tuple[str, str]]:
    """
    配對原始影像與退化影像。
    
    假設檔名規則：
    - 原始影像：image1.bmp, image2.bmp, ...
    - 退化影像：image1_degraded.bmp, image1_noise.bmp, ...
    
    或者按順序配對（前4張為原始，後4張為退化）。
    """
    pairs = []
    
    # 嘗試根據檔名模式配對
    original_images = []
    degraded_images = []
    
    for name in image_names:
        name_lower = name.lower()
        if 'degraded' in name_lower or 'degradation' in name_lower or 'noise' in name_lower or 'blur' in name_lower:
            degraded_images.append(name)
        else:
            original_images.append(name)
    
    # 如果成功分類
    if len(original_images) == len(degraded_images) > 0:
        original_images.sort()
        degraded_images.sort()
        for orig, deg in zip(original_images, degraded_images):
            pairs.append((orig, deg))
        logger.info('Paired images by filename pattern: %d pairs', len(pairs))
    else:
        # 按順序配對（假設前半為原始，後半為退化）
        sorted_names = sorted(image_names)
        half = len(sorted_names) // 2
        if half > 0:
            for i in range(half):
                pairs.append((sorted_names[i], sorted_names[half + i]))
            logger.info('Paired images by order: %d pairs', len(pairs))
        else:
            # 單張影像自己配對（用於測試）
            for name in image_names:
                pairs.append((name, name))
            logger.warning('Could not pair images; using self-pairing for testing')
    
    return pairs


def sanitize_to_uint8(image):
    """Convert float input to uint8 safely."""
    import numpy as np
    return np.clip(np.rint(np.asarray(image, dtype=np.float64)), 0, 255).astype(np.uint8)


def main():
    setup_logging(level='INFO')
    logger = get_logger(__name__)

    try:
        params = RestorationParameters()
        parameter_summary = params.summarize()

        image_directory = resolve_test_image_path(logger)
        image_loader = ImageFileLoader(base_directory_path=image_directory)
        image_names = list(image_loader.list_available_images())
        if not image_names:
            raise FileNotFoundError(
                'No supported test images found. Add bmp/png/jpg/jpeg/tif/tiff files to test_image/. '
                'Execution aborted.'
            )

        # 配對影像
        image_pairs = pair_original_and_degraded_images(image_names, logger)
        logger.info('Found %d image pair(s) for processing', len(image_pairs))

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
                visualizer = ImageRestorationVisualizer()
                loaded_images = image_loader.load_multiple_image_files(image_names)

                processed_items: List[ProcessedItem] = []
                total_pairs = len(image_pairs)

                for index, (original_name, degraded_name) in enumerate(image_pairs, start=1):
                    app.schedule_processing_message(
                        f'Processing {index}/{total_pairs}: {original_name} <-> {degraded_name}')
                    
                    original_image = loaded_images[original_name]
                    degraded_image = loaded_images[degraded_name]
                    
                    logger.info('Processing pair %d/%d: %s <-> %s',
                               index, total_pairs, original_name, degraded_name)
                    
                    # 執行復原處理
                    results = compute_restoration_outputs(degraded_image, params, logger)
                    stats = collect_intensity_statistics(results, original_image, degraded_image)
                    
                    # 計算 PSNR
                    original_uint8 = sanitize_to_uint8(original_image)
                    psnr_inverse = compute_psnr(original_uint8, results.inverse_filtered)
                    psnr_wiener = compute_psnr(original_uint8, results.wiener_filtered)
                    
                    logger.info('  PSNR - Inverse: %.2f dB, Wiener: %.2f dB',
                               psnr_inverse, psnr_wiener)
                    
                    # 儲存結果
                    save_results(f'{os.path.splitext(original_name)[0]}_restored', results, image_loader)
                    
                    # 產生比較圖
                    figure_path = visualize_results(
                        image_filename=original_name,
                        original_image=original_image,
                        degraded_image=degraded_image,
                        results=results,
                        visualizer=visualizer,
                        display_plot_immediately=False,
                    )

                    detail_text = (
                        f"{parameter_summary} | "
                        f"Means: orig={stats['original_mean']:.1f}, deg={stats['degraded_mean']:.1f}, "
                        f"inv={stats['inverse_mean']:.1f}, wien={stats['wiener_mean']:.1f} | "
                        f"PSNR: inv={psnr_inverse:.2f}dB, wien={psnr_wiener:.2f}dB"
                    )

                    processed_items.append(
                        ProcessedItem(
                            filename=f"{original_name} <-> {degraded_name}",
                            comparison_figure_path=figure_path,
                            detail_text=detail_text,
                        )
                    )

                app.schedule_processed_items(processed_items)
                app.schedule_processing_message('Processing complete')

                logger.info('=' * 60)
                logger.info('All restoration operations completed successfully!')
                logger.info('Results saved in results/ directory (BMP outputs and comparison figures).')
                logger.info('Restoration parameters: %s', parameter_summary)
                logger.info('=' * 60)

            except Exception as worker_exception:
                processing_error.set()
                logger.error('Error occurred during processing: %s', worker_exception)
                import traceback
                logger.error(traceback.format_exc())
                app.schedule_error(str(worker_exception))

        threading.Thread(target=processing_worker, daemon=True).start()

        app.run()

        logging.getLogger().removeHandler(gui_handler)
        gui_handler.close()

        if processing_error.is_set():
            sys.exit(1)

    except Exception as main_exception:
        logger.error('Program terminated with errors: %s', main_exception)
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()