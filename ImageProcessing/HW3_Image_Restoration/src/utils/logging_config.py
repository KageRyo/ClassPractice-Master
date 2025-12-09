import logging
import sys
from datetime import datetime


def setup_logging(level=logging.INFO, log_to_file=False, log_filename=None):
    """Configure root logger with console output and optional file sink."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    if log_to_file:
        if log_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'image_restoration_{timestamp}.log'
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logging.info('Logging to file: %s', log_filename)


def get_logger(name):
    """Return a named logger bound to the configured logging setup."""
    return logging.getLogger(name)
