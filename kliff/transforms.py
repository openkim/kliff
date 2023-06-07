from typing import Dict

from kliff.models import Parameter


class Transform:
    def __init__(self, name):
        self.name = name

    def transform(self, inputs, **kwargs):
        raise NotImplementedError

    def inverse(self, inputs, **kwargs):
        raise NotImplementedError

    def __call__(self, inputs, **kwargs):
        return self.transform(inputs, **kwargs)


class ParameterTransform(Transform):
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