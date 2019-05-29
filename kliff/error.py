class ModelError(Exception):
    def __init__(self, msg):
        super(ModelError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg


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
        return repr(self.value) + ' initialization failed'


def report_import_error(name):
    raise ImportError(
        'Optional module "{}" needed by "KLIFF" not found. Please install it '
        'manually.'.format(name)
    )
