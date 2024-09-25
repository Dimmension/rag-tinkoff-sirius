import logging


def setup_logger(filename, level=logging.INFO) -> logging.Logger:
    logging.root.setLevel(level)

    logger = logging.getLogger(filename)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \t %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
