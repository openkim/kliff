from enum import Enum

from kliff._exceptions import TrainerError


class ModelTypes(Enum):
    """
    Enumerates the different types of models that can be used in the training
    process. The different types are:
    - KIM: The KIM model, passed as a string of KIM ID.
    - TORCH: A model implemented in PyTorch, passed as a Python callable.
    - TAR: A model saved in a tar file, which can either be a valid KIM API
        model, where it will be installed in CWD using kim-api collections
        management, or TorchML driver based model, in which case, the TorchScript
        file will be extracted and used. The kind of model is determined from the
        KIM API CMakelists.txt file model drive name string.
    """

    UNDEFINED = -1
    KIM = 0
    TORCH = 1
    TAR = 2

    @staticmethod
    def get_model_enum(input_str: str):
        """
        Get the model type from the input string.

        Args:
            input_str: name of the model type. "kim", "torch", "pt", "pth" or "tar"

        Returns:
            Model type enum.
        """
        if input_str.lower() == "kim":
            return ModelTypes.KIM
        elif (
            input_str.lower() == "torch"
            or input_str.lower() == "pt"
            or input_str.lower() == "pth"
        ):
            return ModelTypes.TORCH
        elif input_str.lower() == "tar":
            return ModelTypes.TAR
        else:
            raise TrainerError(f"Model type {input_str} not supported.")

    @staticmethod
    def get_model_str(input_type):
        """
        Get the model configuration string from the model type.

        Args:
            input_type: input model enum.

        Returns:
            Model configuration string. "KIM", "TORCH" or "TAR"

        """
        if input_type == ModelTypes.KIM:
            return "KIM"
        elif input_type == ModelTypes.TORCH:
            return "TORCH"
        elif input_type == ModelTypes.TAR:
            return "TAR"
        else:
            raise TrainerError(f"Model type {input_type} not supported.")


class DataSource(Enum):
    """
    Enumerates the different types of data sources. The different types are:
    - ASE: ASE atoms objects, or xyz file with configurations. Uses
        ~:class:`~kliff.dataset.Dataset.from_ase` method.
    - COLABFIT: uUses ColabFit dataset exchange instance. Uses
        ~:class:`~kliff.dataset.Dataset.from_colabfit` method.
    - KLIFF: Uses KLIFF compatible extxyz files path. Uses
        ~:class:`~kliff.dataset.Dataset.from_path` method.
    """

    UNDEFINED = -1
    ASE = 0
    COLABFIT = 1
    KLIFF = 2

    @staticmethod
    def get_data_enum(input_str: str):
        """
        Get the data type from the input string.

        Args:
            input_str: name of the data type. "ase", "colabfit" or "kliff"

        Returns:
            Data type enum.

        """
        if input_str.lower() == "ase":
            return DataSource.ASE
        elif input_str.lower() == "colabfit":
            return DataSource.COLABFIT
        elif input_str.lower() == "kliff":
            return DataSource.KLIFF
        else:
            raise TrainerError(f"Data type {input_str} not supported.")

    @staticmethod
    def get_data_str(input_type):
        """
        Get the data configuration string from the data type.

        Args:
            input_type: input data enum.

        Returns:
            Data configuration string. "ASE", "COLABFIT" or "KLIFF"

        """
        if input_type == DataSource.ASE:
            return "ASE"
        elif input_type == DataSource.COLABFIT:
            return "COLABFIT"
        elif input_type == DataSource.KLIFF:
            return "KLIFF"
        else:
            raise TrainerError(f"Data type {input_type} not supported.")


class ConfigurationTransformationTypes(Enum):
    """
    Enumerates the different types of configuration transformations that can be
    applied to the input data. The different types are:
    - GRAPH: Graph based transformation.
    - DESCRIPTORS: Descriptor based transformation.
    - NEIGHBORS: No transformation besides neighbor list computation.
    """

    UNDEFINED = -1
    GRAPH = 0
    DESCRIPTORS = 1
    NEIGHBORS = 2

    @staticmethod
    def get_config_transformation_enum(input_str: str):
        """
        Get the configuration transformation type from the input string.

        Args:
            input_str: name of the configuration transformation type. "graph", "descriptors" or "neighbors"

        Returns:
            Configuration transformation type enum.

        """
        if input_str.lower() == "graph":
            return ConfigurationTransformationTypes.GRAPH
        elif input_str.lower() == "descriptors":
            return ConfigurationTransformationTypes.DESCRIPTORS
        elif input_str.lower() == "neighbors" or input_str.lower() == "none":
            return ConfigurationTransformationTypes.NEIGHBORS
        else:
            raise TrainerError(
                f"Configuration transformation type {input_str} not supported."
            )

    @staticmethod
    def get_config_transformation_str(input_type):
        """
        Get the configuration transformation configuration string from the
        configuration transformation type.

        Args:
            input_type: input configuration transformation enum.

        Returns:
            Configuration transformation configuration string. "GRAPH", "DESCRIPTORS" or "NEIGHBORS"

        """
        if input_type == ConfigurationTransformationTypes.GRAPH:
            return "GRAPH"
        elif input_type == ConfigurationTransformationTypes.DESCRIPTORS:
            return "DESCRIPTORS"
        else:
            raise TrainerError(
                f"Configuration transformation type {input_type} not supported."
            )


class OptimizerProvider(Enum):
    """
    Enumerates the different types of optimizer providers that can be used in the
    training process. The different types are "TORCH" and "SCIPY".
    """

    UNDEFINED = -1
    TORCH = 0
    SCIPY = 1

    @staticmethod
    def get_optimizer_enum(input_str: str):
        """
        Get the optimizer provider from the input string.

        Args:
            input_str: name of the optimizer provider. "torch" or "scipy"

        Returns:
            Optimizer provider enum.

        """
        if input_str.lower() == "torch":
            return OptimizerProvider.TORCH
        elif input_str.lower() == "scipy":
            return OptimizerProvider.SCIPY
        else:
            raise TrainerError(f"Optimizer provider {input_str} not supported.")

    @staticmethod
    def get_optimizer_str(input_type):
        if input_type == OptimizerProvider.TORCH:
            return "TORCH"
        elif input_type == OptimizerProvider.SCIPY:
            return "SCIPY"
        else:
            raise TrainerError(f"Optimizer provider {input_type} not supported.")
