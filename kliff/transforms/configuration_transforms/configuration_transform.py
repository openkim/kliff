from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Union

if TYPE_CHECKING:
    from kliff.dataset import Configuration


class ConfigurationTransform:
    """
    A configuration transform is a function that maps a configuration to a "fingerprint".
    The fingerprint can be any object that represents the configuration, and restriction
    or checks on the fingerprint is not imposed. For example, current configuration transforms
    include graph representations of the configuration,and descriptors.
    """

    def __init__(self, copy_to_config=False):
        self._implicit_fingerprint_copying = copy_to_config

    def forward(self, configuration: "Configuration") -> Any:
        """
        Map a configuration to a fingerprint. Also handle the implicit copying of the
        fingerprint to the configuration.

        Args:
            configuration: Instance of ~:class:`kliff.dataset.Configuration`. For which the
                fingerprint is to be generated.

        Returns:
            Fingerprint of the configuration.
        """
        raise NotImplementedError

    def __call__(self, configuration: "Configuration") -> Any:
        fingerprint = self.forward(configuration)
        if self.copy_to_config:
            configuration.fingerprint(fingerprint)
        return fingerprint

    def inverse(self, *args, **kargs) -> "Configuration":
        """
        Inverse mapping of the transform. This is not implemented for any of the transforms,
        but is there for future use.
        """
        NotImplementedError(
            "Do you mean `backward`?\n"
            "Any of the implemented transforms do not support inverse mapping.\n"
            "For computing jacobian-vector product use `backward` function."
        )

    def transform(self, configuration: "Configuration") -> Any:
        return self(configuration)

    @property
    def copy_to_config(self) -> bool:
        return self._implicit_fingerprint_copying

    @copy_to_config.setter
    def copy_to_config(self, value: bool) -> None:
        self._implicit_fingerprint_copying = value

    def collate_fn(self, config_list: List["Configuration"]) -> List[Any]:
        """
        Collate a list of configurations into a list of transforms. This is useful for
        batch processing.

        Args:
            config_list: List of configurations.
        """
        transforms_list = []
        for conf in config_list:
            transform = self(conf)
            transforms_list.append(transform)
        return transforms_list

    def export_kim_model(self, filename: Path, modelname: str):
        """
        Save the configuration transform to a file.

        Args:
            filename: Name of the file to save the transform to.
            modelname: Name of the model to save.
        """
        raise NotImplementedError


# TODO: should neighbor lists be a transform? It fits the definition as graph.
