from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np

if TYPE_CHECKING:
    from kliff.models.parameter import Parameter


class ParameterTransform:
    """
    Abstract class to transform parameters and inverse transform it back.

    Subclass can implement
        - transform
        - inverse
    """

    def __init__(self, name):
        self.name = name

    def transform(
        self, model_param: Union["Parameter", np.ndarray]
    ) -> Union["Parameter", np.ndarray]:
        raise NotImplementedError

    def inverse_transform(
        self, model_param: Union["Parameter", np.ndarray]
    ) -> Union["Parameter", np.ndarray]:
        raise NotImplementedError

    def __call__(
        self, model_param: Union["Parameter", np.ndarray]
    ) -> Union["Parameter", np.ndarray]:
        return self.transform(model_param)


class LogParameterTransform(ParameterTransform):
    """
    Transform parameters to a log space and transform it back.

    Args:
        param_names: names of the parameters to do the transformation; can be a
            subset of all the parameters.
    """

    def __init__(self):
        super().__init__("log")

    def transform(
        self, model_params: Union["Parameter", np.ndarray]
    ) -> Union["Parameter", np.ndarray]:
        return np.log(model_params)

    def inverse_transform(
        self, model_params: Union["Parameter", np.ndarray]
    ) -> Union["Parameter", np.ndarray]:
        return np.exp(model_params)

    def __call__(
        self, model_params: Union["Parameter", np.ndarray]
    ) -> Union["Parameter", np.ndarray]:
        return self.transform(model_params)
