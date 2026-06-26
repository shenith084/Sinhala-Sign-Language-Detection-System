import logging
import sys

def setup_logger():
    logger = logging.getLogger("SSL400_API")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        ))
        logger.addHandler(handler)
        
    return logger

logger = setup_logger()
