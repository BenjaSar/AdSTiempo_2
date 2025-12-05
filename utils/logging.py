""""Utility module for creating a colored logger with file and console output."""

import logging
import colorlog
import os
import sys

DEBUG = False

def get_logger(name=None, log_file='logs/etl_logging.txt', to_console=DEBUG):
    """Get or create a colored logger instance with file and console output"""
    logger = logging.getLogger(name or 'my_colored_logger')
    
    # Only add handlers if not already present
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"📁 Created logs directory: {log_dir}")
        
        if to_console:
            # Create a ColoredFormatter for console output (CONDITIONAL)
            console_formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(levelname)s:%(name)s:%(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler = colorlog.StreamHandler(stream=sys.stdout)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            print(f"📢 Console logging is ACTIVE.")

        # Create a FileHandler for file output (ALWAYS ACTIVE)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        try:
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            print(f"📝 Logging to file: {log_file}")
        except Exception as e:
            print(f"⚠️  Could not create file handler: {e}")
    
    return logger

if __name__ == "__main__":
    # Create default logger for module-level logging
    logger = get_logger('my_colored_logger')

    # Log messages only on first import
    if not hasattr(logger, '_initialized'):
        logger.debug("This is a debug message.")
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")
        logger.error("This is an error message.")
        logger.critical("This is a critical message.")
        logger._initialized = True
