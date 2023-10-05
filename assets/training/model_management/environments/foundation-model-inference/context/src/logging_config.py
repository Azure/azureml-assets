import logging
import sys


def configure_logger(name):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    format_str = (
        "%(asctime)s [%(module)s] %(funcName)s "
        "%(lineno)s: %(levelname)-8s [%(process)d] %(message)s"
    )
    formatter = logging.Formatter(format_str)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
