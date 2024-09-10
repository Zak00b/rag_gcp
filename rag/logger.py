import colorlog
import logging

handler = colorlog.StreamHandler()

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)

# Set the log level to info
logger.setLevel(logging.INFO)

# Add the following lines to set green color for info level logs
handler.setFormatter(colorlog.ColoredFormatter('%(green)s%(levelname)s:%(name)s:%(message)s'))

