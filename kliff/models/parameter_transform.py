from typing import Dict, List

import numpy as np

from kliff.models.parameter import Parameter


class ParameterTransform:
    """
    Abstract class to transform parameters and inverse transform it back.

    Subclass can implement
        - transform
        - inverse_transform
    """

    def transform(self, model_params: Dict[str, Parameter]) -> Dict[str, Parameter]:
        return model_params

    def inverse_transform(
        self, model_params: Dict[str, Parameter]
    ) -> Dict[str, Parameter]:
        return model_params

    def __call__(self, model_params: Dict[str, Parameter]) -> Dict[str, Parameter]:
        return self.transform(model_params)


class LogParameterTransform(ParameterTransform):
    """
    Transform parameters to a log space and transform it back.

    Args:
        param_names: names of the parameters to do the transformation; can be a
            subset of all the parameters.
    """

    def __init__(self, param_names: List[str]):
        super().__init__()
        self.param_names = param_names

    def transform(self, model_params: Dict[str, Parameter]) -> Dict[str, Parameter]:
        for name in self.param_names:
            p = model_params[name]
            p_val = np.asarray(p.value)

            assert np.asarray(p_val).min() >= 0, (
                f"Cannot log transform parameter `{name}`, got negative values: "
                f"{p_val}"
            )
            p.value = np.log(p_val)

        return model_params

    def inverse_transform(
        self, model_params: Dict[str, Parameter]
    ) -> Dict[str, Parameter]:
        for name in self.param_names:
            p = model_params[name]
            p_val = np.asarray(p.value)
            p.value = np.exp(p_val)

        return model_params

    def __call__(self, model_params: Dict[str, Parameter]) -> Dict[str, Parameter]:
        return self.transform(model_params)
