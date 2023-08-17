import logging

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)

def log(level: logging._Level, message: str):
    logging.log(level=level, msg=message)

def info(message: str):
    log(logging.INFO, message)

def warning(message: str):
    log(logging.WARNING, message)

def error(message: str):
    log(logging.ERROR, message)
