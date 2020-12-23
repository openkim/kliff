class InputError(Exception):
    def __init__(self, msg):
        super(InputError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg


class KeyNotFoundError(Exception):
    def __init__(self, msg):
        super(KeyNotFoundError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg


class SupportError(Exception):
    def __init__(self, msg):
        super(SupportError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg


class InitializationError(Exception):
    def __init__(self, value):
        self.value = value

    def __expr__(self):
        return repr(self.value) + " initialization failed"


def report_import_error(package, classname=None):
    if classname is not None:
        msg = f"To use `{classname}`, `{package}` is need. Please install it first."
    else:
        msg = f"Package `{package}` needed by KLIFF not found. Please install it first."

    raise ImportError(msg)
