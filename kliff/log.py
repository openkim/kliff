import logging
import warnings


def set_up_logger():
    logging.basicConfig(
        filename="kliff.log",
        format="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
        level=logging.INFO,
    )


def log_entry(logger, message, level="info", print_end="\n", warning_category=Warning):
    logger_level = getattr(logger, level)
    logger_level(message)

    if level == "info":
        print(message, end=print_end)
    elif level == "warning":
        warnings.warn(message, category=warning_category)
