import os
import logging
import warnings


class Logger:
    def __init__(self, level='warning', filename='kliff.log'):

        # remove log file if existing
        if os.path.exists(filename):
            os.remove(filename)

        logging.basicConfig(
            filename=filename, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s'
        )

        self.level = level.upper()
        self.filename = filename
        self.loggers = []

    def set_level(self, level):
        self.level = level.upper()
        # reset level for previous instantiated loggers
        for l in self.loggers:
            l.setLevel(self.level)

    def get_logger(self, name=None):
        if name is None:
            logger = logging.getLogger()
        else:
            logger = logging.getLogger(name)
        logger.setLevel(self.level)

        self.loggers.append(logger)
        return logger


def log_entry(logger, message, level='info', print_end='\n', warning_category=Warning):
    logger_level = getattr(logger, level)
    logger_level(message)

    if level == 'info':
        print(message, end=print_end)
    elif level == 'warning':
        warnings.warn(message, category=warning_category)
