"""
This module contains exceptions to be raised in kliff modules, along with details on
where they are raised.
"""


class TrainerError(Exception):
    """
    Exceptions to be raised in Trainer and associated classes.
    """

    def __init__(self, message):
        super().__init__(message)
