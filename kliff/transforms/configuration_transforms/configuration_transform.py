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
            configuration.fingerprint(fingerprint)
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
