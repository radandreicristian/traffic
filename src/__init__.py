import logging
import sys

logger = logging.getLogger('traffic')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%d %b, %H:%M")
stdout_handler = logging.StreamHandler(sys.stdout)

logger.addHandler(stdout_handler)
