from math import log, exp


class ParameterTransform:
    """
    Abstract class to transform parameters and inverse transform it back.

    Subclass can implement
        - transform
        - inverse
    """

    def __init__(self, name):
        super().__init__(name)

    def transform(self, model_params):
        raise NotImplementedError

    def inverse(self, model_params):
        raise NotImplementedError

    def __call__(self, model_params):
        return self.transform(model_params)


class LogParameterTransform(ParameterTransform):
    """Natural log transformation.
    This is an example implementation of Parameter Transform.
    """
    def __init__(self):
        self.name = "log"
        super().__init__(self.name)

    def transform(self, model_params):
        return log(model_params)

    def inverse(self, model_params):
        return exp(model_params)
