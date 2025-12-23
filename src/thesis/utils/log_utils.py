
import logging
import sys

def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger with a standard format.
    
    :param name: Name of the logger
    :param level: Logging level (default: logging.INFO)
    :return: Configured Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger already has handlers, assume it's set up and return dependencies
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger
