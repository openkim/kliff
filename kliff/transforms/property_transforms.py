from typing import TYPE_CHECKING, List, Union

import numpy as np

from kliff.dataset import Configuration, Dataset


class PropertyTransform:
    def __int__(self):
        self.original_property_value_map = None
        self.property_key = "energy"

    def transform(self, dataset: Union[List[Configuration], Dataset]):
        raise PropertyTransformError("This method is not implemented.")

    def inverse(self, dataset: Union[List[Configuration], Dataset]):
        raise PropertyTransformError("This method is not implemented.")

    def __call__(self, dataset: Union[List[Configuration], Dataset]):
        self.transform(dataset)

    @staticmethod
    def get_configuration_list(
        dataset: Union[List[Configuration], Dataset]
    ) -> List[Configuration]:
        configuration_list = []
        if isinstance(dataset, Dataset):
            configuration_list = dataset.get_configs()
        elif isinstance(dataset, List):
            configuration_list = dataset
        else:
            PropertyTransformError("Unknown format of dataset.")
        return configuration_list


class NormalizedPropertyTransform(PropertyTransform):
    def __init__(self, property_key="energy", keep_original=True):
        self.original_property_value_map = None
        self.property_key = property_key
        self.keep_original = keep_original
        self.mean = 0.0
        self.std = 1.0

    def transform(self, dataset: Union[List[Configuration], Dataset]):
        configuration_list = self.get_configuration_list(dataset)
        n_configs = len(configuration_list)
        original_property_values = list(
            map(lambda config: getattr(config, self.property_key), configuration_list)
        )
        original_property_values = np.vstack(original_property_values)
        if self.keep_original:
            self.original_property_value_map = original_property_values
        self.mean = original_property_values.mean()
        self.std = original_property_values.std()
        for configuration in configuration_list:
            property_to_transform = getattr(configuration, self.property_key)
            property_to_transform -= self.mean
            property_to_transform /= self.std
            setattr(configuration, self.property_key, property_to_transform)

    def inverse(self, dataset: Union[List[Configuration], Dataset]):
        configuration_list = self.get_configuration_list(dataset)
        n_configs = len(configuration_list)
        for configuration in configuration_list:
            property_to_transform = getattr(configuration, self.property_key)
            property_to_transform *= self.std
            property_to_transform += self.mean
            setattr(configuration, self.property_key, property_to_transform)


class RMSNormalizePropertyTransform(PropertyTransform):
    def __init__(self, property_key="forces", keep_original=False):
        self.original_property_value_map = None
        self.property_key = property_key
        self.keep_original = keep_original
        self.rms_mean = 0.0

    def transform(self, dataset: Union[List[Configuration], Dataset]):
        configuration_list = self.get_configuration_list(dataset)
        n_configs = len(configuration_list)
        original_property_values = list(
            map(lambda config: getattr(config, self.property_key), configuration_list)
        )
        original_property_values = np.vstack(original_property_values)
        if self.keep_original:
            self.original_property_value_map = original_property_values
        self.rms_mean = np.sqrt(np.mean(np.square(original_property_values)))
        for configuration in configuration_list:
            property_to_transform = getattr(configuration, self.property_key)
            property_to_transform /= self.rms_mean
            setattr(configuration, self.property_key, property_to_transform)

    def inverse(self, dataset: Union[List[Configuration], Dataset]):
        configuration_list = self.get_configuration_list(dataset)
        n_configs = len(configuration_list)
        for configuration in configuration_list:
            property_to_transform = getattr(configuration, self.property_key)
            property_to_transform *= self.rms_mean
            setattr(configuration, self.property_key, property_to_transform)


class RMSMagnitudeNormalizePropertyTransform(PropertyTransform):
    def __init__(self, property_key="forces", keep_original=False):
        self.original_property_value_map = None
        self.property_key = property_key
        self.keep_original = keep_original
        self.rms_mean_magnitude = 0.0

    def transform(self, dataset: Union[List[Configuration], Dataset]):
        configuration_list = self.get_configuration_list(dataset)
        n_configs = len(configuration_list)
        original_property_values = list(
            map(lambda config: getattr(config, self.property_key), configuration_list)
        )
        original_property_values = np.vstack(original_property_values)
        if self.keep_original:
            self.original_property_value_map = original_property_values
        self.rms_mean_magnitude = np.sqrt(
            np.mean(np.sum(np.square(original_property_values), 1))
        )
        for configuration in configuration_list:
            property_to_transform = getattr(configuration, self.property_key)
            property_to_transform /= self.rms_mean_magnitude
            setattr(configuration, self.property_key, property_to_transform)

    def inverse(self, dataset: Union[List[Configuration], Dataset]):
        configuration_list = self.get_configuration_list(dataset)
        n_configs = len(configuration_list)
        for configuration in configuration_list:
            property_to_transform = getattr(configuration, self.property_key)
            property_to_transform *= self.rms_mean_magnitude
            setattr(configuration, self.property_key, property_to_transform)


class PropertyTransformError(Exception):
    def __init__(self, msg):
        super(PropertyTransformError, self).__init__(msg)
        self.msg = msg
