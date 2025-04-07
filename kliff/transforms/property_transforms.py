from typing import List, Union

import numpy as np

from kliff.dataset import Configuration, Dataset


class PropertyTransform:
    """
    A property transform is a function that maps a property of a configuration to a
    transformed property. For example, current property transforms include normalization
    of the energy and forces.

    Attributes:
        keep_original: If True, the original property values are kept in the configuration
            object. Otherwise, the original property values are discarded.
        original_values: A map that stores the original property values of
            configurations. It stores list of property values for each configuration.
        property_key: The key of the property to be transformed.
    """

    def __init__(self, property_key: str = "energy", keep_original: bool = False):
        self._keep_original = keep_original
        self.original_values = None
        self.property_key = property_key

    @property
    def keep_original(self):
        return self._keep_original

    def transform(self, dataset: Union[List[Configuration], Dataset]):
        """
        Transform the property of a configuration.

        Args:
            dataset: A list of configurations or a dataset.
        """
        raise PropertyTransformError("This method is not implemented.")

    def inverse(self, dataset: Union[List[Configuration], Dataset]):
        """
        Inverse transform the property of a configuration.

        Args:
            dataset: A list of configurations or a dataset.
        """
        raise PropertyTransformError("This method is not implemented.")

    def __call__(self, dataset: Union[List[Configuration], Dataset]):
        self.transform(dataset)

    @staticmethod
    def get_configuration_list(
        dataset: Union[List[Configuration], Dataset]
    ) -> List[Configuration]:
        """
        Get a list of configurations from a dataset. This method ensures constant API
        for any arbitrary dataset. It is expected that this method will maintain the
        order of the Configurations in the dataset. Tihs enqures the inverse property
        mapping is done correctly.

        Args:
            dataset: A list of configurations or a dataset.

        Returns:
            A list of configurations.
        """
        configuration_list = []
        if isinstance(dataset, Dataset):
            configuration_list = dataset.get_configs()
        elif isinstance(dataset, List):
            configuration_list = dataset
        else:
            PropertyTransformError("Unknown format of dataset.")
        return configuration_list

    def _get_property_values(
        self, dataset: Union[List[Configuration], Dataset]
    ) -> np.ndarray:
        """
        Get the property values from all the configurations in a dataset and return
        them as a numpy array.

        Args:
            dataset: A list of configurations or a dataset

        Returns:
            A numpy array of property values.
        """
        configuration_list = self.get_configuration_list(dataset)
        property_values = list(
            map(lambda config: getattr(config, self.property_key), configuration_list)
        )
        return np.asarray(property_values)

    def _set_property_values(
        self,
        dataset: Union[List[Configuration], Dataset],
        property_values: Union[np.ndarray, List[Union[float, int]]],
    ):
        """
        Set the property values of all the configurations in a dataset. This method
        assumes that the transformed values are in the same order as the configurations
        and can be iterated over using sinle integer index.
        Args:
            dataset: A list of configurations or a dataset
            property_values: A numpy array of property values.
        """
        configuration_list = self.get_configuration_list(dataset)
        for i, configuration in enumerate(configuration_list):
            setattr(configuration, self.property_key, property_values[i])


class NormalizedPropertyTransform(PropertyTransform):
    """
    Normalize the property of a configuration to zero mean and unit variance.
    .. math::
        x' = \\frac{x - \\mu}{\\sigma}
    """

    def __init__(self, property_key: str = "energy", keep_original: bool = False):
        super().__init__(property_key, keep_original)
        self.original_values = None
        self.property_key = property_key
        self._keep_original = keep_original
        self.mean = 0.0
        self.std = 1.0

    def transform(self, dataset: Union[List[Configuration], Dataset]):
        original_property_values = self._get_property_values(dataset)
        if self.keep_original:
            self.original_values = original_property_values
        self.mean = original_property_values.mean()
        self.std = original_property_values.std()
        transformed_values = (original_property_values - self.mean) / self.std
        self._set_property_values(dataset, transformed_values)

    def inverse(self, dataset: Union[List[Configuration], Dataset]):
        transformed_values = self._get_property_values(dataset)
        transformed_values = transformed_values * self.std + self.mean
        self._set_property_values(dataset, transformed_values)


class RMSNormalizePropertyTransform(PropertyTransform):
    """
    Normalize the property of a configuration to using the root mean square of the property.
    It is useful for normalizing oscillators properties, usually because mean for
    such properties is zero.

    .. math::
        x' = \\frac{x}{\\sqrt{\\frac{1}{N}\\sum_{i=1}^N x_i^2}}
    """

    def __init__(self, property_key: str = "forces", keep_original: bool = False):
        super().__init__(property_key, keep_original)
        self.original_values = None
        self.property_key = property_key
        self._keep_original = keep_original
        self.rms_mean = 0.0

    def transform(self, dataset: Union[List[Configuration], Dataset]):
        original_property_values = self._get_property_values(dataset)
        if self.keep_original:
            self.original_values = original_property_values
        self.rms_mean = np.sqrt(np.mean(np.square(original_property_values)))
        transformed_values = original_property_values / self.rms_mean
        self._set_property_values(dataset, transformed_values)

    def inverse(self, dataset: Union[List[Configuration], Dataset]):
        transformed_values = self._get_property_values(dataset)
        transformed_values = transformed_values * self.rms_mean
        self._set_property_values(dataset, transformed_values)


class RMSMagnitudeNormalizePropertyTransform(PropertyTransform):
    """
    Normalize the property of a configuration using the root mean square of the magnitude
    of the property. This method is useful for normalizing forces.

    .. math::
        x' = \\frac{x}{\\sqrt{\\frac{1}{N}\\sum_{i=1}^N \\|x_i\\|^2}}
    """

    def __init__(self, property_key: str = "forces", keep_original: bool = False):
        super().__init__(property_key, keep_original)
        self.original_values = None
        self.property_key = property_key
        self._keep_original = keep_original
        self.rms_mean_magnitude = 0.0

    def transform(self, dataset: Union[List[Configuration], Dataset]):
        original_property_values = self._get_property_values(dataset)
        if self.keep_original:
            self.original_values = original_property_values
        self.rms_mean_magnitude = np.sqrt(
            np.mean(np.square(np.linalg.norm(original_property_values, axis=1)))
        )
        transformed_values = original_property_values / self.rms_mean_magnitude
        self._set_property_values(dataset, transformed_values)

    def inverse(self, dataset: Union[List[Configuration], Dataset]):
        transformed_values = self._get_property_values(dataset)
        transformed_values = transformed_values * self.rms_mean_magnitude
        self._set_property_values(dataset, transformed_values)


class PropertyTransformError(Exception):
    def __init__(self, msg):
        super(PropertyTransformError, self).__init__(msg)
        self.msg = msg
