import logging


class Logger(object):
    def __init__(self, level='warning', filename='kliff.log'):
        self.level = level.upper()
        self.filename = filename
        logging.basicConfig(
            filename=self.filename,
            format='%(asctime)s:%(name)s:%(levelname)s: %(message)s',
        )
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
