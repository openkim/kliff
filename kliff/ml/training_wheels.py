import numpy as np
import torch
from .graphs import KIMTorchGraphGenerator, KIMTorchGraph
from torch_scatter import scatter
from .descriptors import Descriptor
import os

# TODO: Figure out way to incorporate original descriptor class
# you will need to take manual jacobian vector product

class TorchDescriptorFunction(torch.autograd.Function):
    """
    This class is a wrapper for the Descriptor class in kliff_torch.
    It is used to compute the descriptor values and derivatives.
    """
    @staticmethod
    def forward(ctx, DescFunCtx, configuration, coordinates):
        """
        Coordinates tensor needs to be passed separately .
        Args:
        :param ctx:
        :param DescFunCtx:
        :param coordinates: Coordinate tensor to accumulate gradients

        Returns:

        """
        ctx.DescFunCtx = DescFunCtx
        ctx.configuration = configuration
        descriptor_tensor = DescFunCtx.forward(configuration)
        descriptor_tensor = torch.from_numpy(descriptor_tensor)
        descriptor_tensor.requires_grad_(True)
        return descriptor_tensor

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Args:
        :param ctx:
        :param grad_outputs:
        :return:
        """
        DescFunCtx = ctx.DescFunCtx
        configuration = ctx.configuration
        dE_dzeta = grad_outputs.numpy()
        dE_dr = DescFunCtx.backward(configuration, dE_dzeta)
        dE_dr = torch.from_numpy(dE_dr)
        dE_dr.requires_grad_(True)
        return None, None, dE_dr


class TorchLegacyDescriptorFunction(torch.autograd.Function):
    """
    This class is a wrapper for the Descriptor class in kliff legacy descriptors.
    """
    @staticmethod
    def forward(ctx, DescFunCtx, configuration, coordinates):
        """
        Coordinates tensor needs to be passed separately .
        Args:
        :param ctx:
        :param DescFunCtx:
        :param coordinates: Coordinate tensor to accumulate gradients

        Returns:

        """
        ctx.DescFunCtx = DescFunCtx
        ctx.configuration = configuration
        descriptor_tensor, DescJacobian, _ = DescFunCtx.transform(configuration, fit_forces=True)
        ctx.DescJacobian = DescJacobian
        descriptor_tensor = torch.from_numpy(descriptor_tensor)
        descriptor_tensor.requires_grad_(True)
        return descriptor_tensor

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Args:
        :param ctx:
        :param grad_outputs:
        :return:
        """
        DescFunCtx = ctx.DescFunCtx
        configuration = ctx.configuration
        dE_dzeta = grad_outputs.numpy()
        dE_dr = np.einsum('ijk,ij->i', ctx.DescJacobian, dE_dzeta)
        dE_dr = torch.from_numpy(dE_dr.reshape(-1, 3))
        dE_dr.requires_grad_(True)
        return None, None, dE_dr


class TorchGraphFunction(torch.autograd.Function):
    """
    This class is a wrapper for the KIMTorchGraph class in kliff_torch.
    """
    @staticmethod
    def forward(ctx, GraphCtx:KIMTorchGraphGenerator, configuration, coordinate:torch.Tensor):
        """
        Coordinates tensor needs to be passed separately .
        :param ctx:
        :param GraphCtx:
        :param configuration:
        :param coordinate:
        :return:
        """
        ctx.GraphCtx = GraphCtx
        graph:KIMTorchGraph = GraphCtx.generate_graph(configuration)
        outputs = [graph.species, graph.coords]
        for i in range(graph.n_layers):
            outputs.append(graph.__getattr__(f"edge_index{i}"))
        outputs.append(graph.contributions)
        ctx.graph = graph
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Args:
        :param ctx:
        :param grad_outputs:
        :return:
        """
        graph = ctx.graph
        images = graph.images
        d_coords = grad_outputs[1]
        d_coords = scatter(d_coords, images, 0)
        return None, None, d_coords


class TrainingWheels(torch.nn.Module):
    """
    Training wheels acts as an adopter between your models, and various numpy components, such as graphs, descriptors,
    etc. This will wrap models and work as a seamless interface to torch optimizer.
    """
    @classmethod
    def init_graph(cls, model, cutoff, n_layers, species, model_returns_forces=False):
        """
        Initialize a graph generator context
        :param model:
        :param cutoff:
        :param n_layers:
        :param species:
        :param model_returns_forces:
        :return:
        """
        kgg = KIMTorchGraphGenerator(species, cutoff, n_layers,as_torch_geometric_data=True)
        preprocessor = TorchGraphFunction()
        return cls(model, preprocessor, generator_ctx=kgg,
                   model_returns_forces=model_returns_forces)

    @classmethod
    def init_descriptor(cls, model, cutoff, descriptor_kind, species, hyperparams, model_returns_forces=False,
                        use_libdescriptor=True, **kwargs):
        """
        Initialize a descriptor context
        :param model:
        :param cutoff:
        :param descriptor_kind:
        :param species:
        :param model_returns_forces:
        :param use_libdescriptor:
        :param kwargs:
        :return:
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
        Initialize a descriptor context
        :param model:
        :param descriptor:
        :param model_returns_forces:
        :param use_libdescriptor:
        :return:
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

        :param configuration:
        :return:
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
        with open(f"{model_name}/CMakeLists.txt", "w") as f:
            f.write("cmake_minimum_required(VERSION 3.10)\n\n")
            f.write("list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_CMAKE_PREFIX_DIR})\n")
            f.write("find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)\n")
            f.write('kim_api_items_setup_before_project(ITEM_TYPE "portableModel")\n\n')

            f.write(f"project({model_name})\n\n")
            f.write(f'kim_api_items_setup_after_project(ITEM_TYPE "portableModel")\n')

            f.write('add_kim_api_model_library(\n')
            f.write('  NAME            ${PROJECT_NAME}\n')
            f.write('  DRIVER_NAME     "TorchMLModelDriver"\n')
            f.write('  PARAMETER_FILES  ')
            for file in file_names:
                f.write(f' "{file}" ')
            f.write(')\n')
