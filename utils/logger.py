import logging
import sys
from config.settings import LOG_FILE_PATH

def setup_logger(name: str = "rag_assistant") -> logging.Logger:
    """Sets up a logger that logs to both console and a log file."""
    logger = logging.getLogger(name)
    
    # If logger already has handlers, don't add more (prevent duplicates)
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    try:
        file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to create file handler for logging: {e}", file=sys.stderr)
        
    return logger

# Default logger instance
logger = setup_logger()
