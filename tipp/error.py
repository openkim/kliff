class InputError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return self.value


class SupportError(Exception):
  def __init__(self, value):
     self.value = value
  def __str__(self):
    return repr(self.value) + ' computation not supported by model'


class InitializationError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value) + ' initialization failed'


class KIMCalculatorError(Exception):
  def __init__(self, msg):
    self.msg = msg
  def __str__(self):
    return reprself.msg

def check_error(error, msg):
  if error != 0 and error is not None:
    raise KIMCalculatorError('Calling "{}" failed.\n'.format(msg))

def report_error(msg):
  raise KIMCalculatorError(msg)

