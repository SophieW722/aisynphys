import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Set up default logging to stderr
stderr_log_handler = logging.StreamHandler()
stderr_log_handler.setLevel(logging.INFO)
logger.addHandler(stderr_log_handler)



