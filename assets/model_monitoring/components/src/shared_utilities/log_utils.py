import logging

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)

def log(level: int, message: str):
    logging.log(level=level, msg=message)

def log_info(message: str):
    log(logging.INFO, message)

def log_warning(message: str):
    log(logging.WARNING, message)

def log_error(message: str):
    log(logging.ERROR, message)
