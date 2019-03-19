class CalculatorError(Exception):
    def __init__(self, msg):
        super(CalculatorError, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg


class InputError(Exception):
    def __init__(self, msg):
        super(InputError, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg


class KeyNotFoundError(Exception):
    def __init__(self, msg):
        super(KeyNotFoundError, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg


class SupportError(Exception):
    def __init__(self, msg):
        super(Support, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg


class InitializationError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value) + ' initialization failed'
