import logging
import sys

formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%s")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.NOTSET)
stdout_handler.setFormatter(formatter)

logging.basicConfig(handlers=[stdout_handler], level=logging.INFO)
