from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from kliff.dataset import Configuration


class ConfigurationTransform:
    def __init__(self, implicit_fingerprint_copying=False):
        self.implicit_fingerprint_copying = implicit_fingerprint_copying

    def forward(self, configuration: "Configuration"):
        raise NotImplementedError

    def __call__(self, configuration: "Configuration"):
        fingerprint = self.forward(configuration)
        if self.implicit_fingerprint_copying:
            configuration.set_fingerprint(fingerprint)
        else:
            return fingerprint

    def inverse(self, *args, **kargs):
        NotImplementedError(
            "Do you mean `backward`?\n"
            "Any of the implemented transforms do not support inverse mapping.\n"
            "For computing jacobian-vector product use `backward` function."
        )

    def transform(self, configuration: "Configuration"):
        return self.__call__(configuration)

    def set_implicit_fingerprinting(self, value: bool):
        self.implicit_fingerprint_copying = value

    def collate_fn(self, config_list):
        """
        Collate function for use with a Pytorch DataLoader.
        :param config_list:
        :return: transforms
        """
        transforms_list = []
        for conf in config_list:
            transform = self.forward(conf)
            transforms_list.append(transform)
        return transforms_list


# TODO: should neighbor lists be a transform? It fits the defination as graph.
