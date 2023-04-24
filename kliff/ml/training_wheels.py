import numpy as np
import torch
from .graphs import KIMTorchGraphGenerator, KIMTorchGraph
from .descriptors import Descriptor
import os

try:
    import numdifftools as nd
    num_diff_available = True
except ImportError:
    num_diff_available = False

from .grad import *

# TODO: Figure out way to incorporate original descriptor class
# you will need to take manual jacobian vector product

class TrainingWheels(torch.nn.Module):
    """
    Training wheels acts as an adopter between your models, and various numpy components, such as graphs, descriptors,
    etc. This will wrap models and work as a seamless interface to torch optimizer.
    """
    @classmethod
    def init_graph(cls, model, cutoff, n_layers, species, model_returns_forces=False):
        """
        Initialize a graph generator context and a TrainingWheels object.
        :param model: TorchScript model to be trained
        :param cutoff: Cutoff radius
        :param n_layers: Number of convolution layers
        :param species: List of atomic species
        :param model_returns_forces: Whether the model is a gradient domain model
        :return: :class:`kliff.ml.training_wheels.TrainingWheels`
        """
        kgg = KIMTorchGraphGenerator(species, cutoff, n_layers,as_torch_geometric_data=True)
        preprocessor = TorchGraphFunction()
        return cls(model, preprocessor, generator_ctx=kgg,
                   model_returns_forces=model_returns_forces)

    @classmethod
    def init_descriptor(cls, model, cutoff, descriptor_kind, species, hyperparams, model_returns_forces=False,
                        use_libdescriptor=True, **kwargs):
        """
        Initialize a TrainingWheels object along with a descriptor context. The generated descriptor context will be
        passed to the forward function. The descriptor context is initialized with the given hyperparameters if
        `use_libdescriptor` is True, otherwise the hyperparameters are passed to the descriptor context as kwargs.
        This for flexibility of using different descriptor implementations.
        :param model: TorchScript model to be trained
        :param cutoff: Cutoff radius
        :param descriptor_kind: Descriptor kind name obtained from :func:`kliff.ml.descriptor.show_available_descriptors()`
        :param species: List of atomic species
        :param model_returns_forces: Whether the model is a gradient domain model
        :param use_libdescriptor: Whether to use the descriptor implementation in libdescriptor
        :param kwargs: Hyperparameters for custom descriptor context
        :return: :class:`kliff.ml.training_wheels.TrainingWheels` instance
        """
        if use_libdescriptor:
            descriptor_ctx = Descriptor(cutoff, species, descriptor_kind, hyperparams)
        else:
            descriptor_ctx = Descriptor(cutoff, descriptor_kind, **kwargs)
        preprocessor = TorchDescriptorFunction()
        return cls(model, preprocessor, generator_ctx=descriptor_ctx,
                   model_returns_forces=model_returns_forces)

    @classmethod
    def from_descriptor_instance(cls, model, descriptor, model_returns_forces=False):
        """
        Initialize a descriptor context from  an initialized descriptor context . The generated descriptor
        context will be passed to the forward function.
        :param model: TorchScript model to be trained
        :param descriptor: :class:`kliff.ml.Descriptor` instance
        :param model_returns_forces: Whether the model is a gradient domain model
        :return: :class:`kliff.ml.training_wheels.TrainingWheels` instance
        """
        descriptor_ctx = descriptor
        preprocessor = TorchDescriptorFunction()
        return cls(model, preprocessor, generator_ctx=descriptor_ctx,
                   model_returns_forces=model_returns_forces)

    def __init__(self, model, preprocessor=None, generator_ctx=None, model_returns_forces=False):
        super(TrainingWheels, self).__init__()
        if preprocessor:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = lambda x: x
        self.generator_ctx = generator_ctx
        self.model = model
        self.parameters = model.parameters()
        self.model_returns_forces = model_returns_forces

    def forward(self, configuration):
        """
        Forward pass of the model. This will take in a :class:`kliff.dataset.Configuration` object, and return a
        dictionary containing the energy, forces, and stress. The forces are negative of the gradient returned by the
        model. As it returns forces as well, no backpropagation is needed.
        :param configuration: :class:`kliff.dataset.Configuration` object or :class:`ase.Atoms` object
        :return: Dictionary containing energy, forces, and stress
        """
        # coordinate_tensor = torch.from_numpy(self.descriptor_ctx.get_padded_coordinates(configuration))
        coordinate_tensor = torch.from_numpy(configuration.coords)
        coordinate_tensor.requires_grad_(True)
        model_inputs = self.preprocessor.apply(self.generator_ctx, configuration, coordinate_tensor)
        if self.model_returns_forces:
            energy, forces = self.model(*model_inputs)
            energy = energy.sum()
        else:
            if isinstance(model_inputs, tuple):
                energy = self.model(*model_inputs)
            else:
                energy = self.model(model_inputs)
            energy = energy.sum()
            forces, = torch.autograd.grad([energy], [coordinate_tensor], retain_graph=True, allow_unused=True)
        return {"energy": energy, "forces": -forces, "stress": None}

    def get_parameters(self):
        return self.parameters

    def save_kim_model(self, model_name):
        """
        Save the TorchScript model as the KIM ported model. It will also save the descriptor and graph
        information if applicable. The model will be saved in the model_name directory. The model will be saved as
        model.pt, and the descriptor and graph information will be saved as kim_model.param and kim_descriptor.dat
        respectively. The saved model can be installed as
        .. code-block:: bash
            kim-api-collections-management install user/system  model_name

        :param model_name: Model name (Recommended format: MD_000000000000_000)
        :return:
        """
        try:
            os.mkdir(model_name)
        except FileExistsError:
            pass
        model_jit = torch.jit.script(self.model)
        model_jit.save(os.path.join(model_name, "model.pt"))

        if type(self.generator_ctx) == KIMTorchGraphGenerator:
            self.generator_ctx.save_kim_model(model_name, "model.pt")
            file_names = ["kim_model.param", "model.pt"]
        elif type(self.generator_ctx) == Descriptor:
            self.generator_ctx.write_kim_params(model_name, "kim_descriptor.dat")
            self.generator_ctx.save_kim_model(model_name, "model.pt")
            file_names = ["kim_model.param","kim_descriptor.dat", "model.pt"]
        else:
            raise NotImplementedError("Only graph and descriptor context are supported.")
        self._write_cmake_file(model_name, file_names)

    @staticmethod
    def _write_cmake_file(model_name, file_names):
        """
        Write a cmake file for the TorchMLModelDriver
        :param model_name: string containing the name of the model
        :param file_names: name of the files to to be included in model
        :return:
        """
        with open(f"{model_name}/CMakeLists.txt", "w") as f:
            f.write("cmake_minimum_required(VERSION 3.10)\n\n")
            f.write("list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_CMAKE_PREFIX_DIR})\n")
            f.write("find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)\n")
            f.write('kim_api_items_setup_before_project(ITEM_TYPE "portableModel")\n\n')

            f.write(f"project({model_name})\n\n")
            f.write(f'kim_api_items_setup_after_project(ITEM_TYPE "portableModel")\n')

            f.write('add_kim_api_model_library(\n')
            f.write('  NAME            ${PROJECT_NAME}\n')
            f.write('  DRIVER_NAME     "TorchMLMD__MD_000000000000_000"\n')
            f.write('  PARAMETER_FILES  ')
            for file in file_names:
                f.write(f' "{file}" ')
            f.write(')\n')
