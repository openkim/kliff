import numpy as np
from loguru import logger
from torch_scatter import scatter

try:
    import torch
    from torch.nn import Parameter, Module
except ImportError:
    torch = None
    Parameter = None
    Module = None
    logger.warning("Torch is not installed. OptimizerTorch will not work correctly.")

try:
    import numdifftools as nd
    num_diff_available = True
except ImportError:
    num_diff_available = False


class TorchDescriptorFunction(torch.autograd.Function):
    """
    This class is a :class:`torch.autograd.Function` wrapper for the Descriptor class :class:`kliff.ml.Descriptor`.
    It is used to compute the descriptor values and derivatives in a PyTorch compatible API. It performs the
    vector-Jacobian product for the descriptor values and derivatives, and return gradients w.r.t positions.
    """
    @staticmethod
    def forward(ctx, DescFunCtx, configuration, coordinates):
        """
        Forward function for computing the descriptors. Coordinates tensor needs to be passed separately .
        Args:
        :param ctx: Context object of the autograd function
        :param DescFunCtx: Descriptor context object :class:`kliff.ml.Descriptor`
        :param coordinates: Coordinate tensor to accumulate gradients in the backward pass (`coordinates.grad`)

        Returns: Descriptor tensor

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
        Backward function for computing the gradients w.r.t positions.
        Args:
        :param ctx: Context object of the autograd function
        :param grad_outputs: Input tensors for computing vector-Jacobian product
        :return: Gradients w.r.t positions
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
    This class is a wrapper for the Descriptor class in kliff legacy descriptors :class:`kliff.legacy.descriptors.Descriptor`.
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


if num_diff_available:
    # KIMModel wrappers, there will two kind of wrappers, one which computes the dataset,
    # and the others which accept parameters and configurations as input.
    class TorchKIMModelWrapper(torch.autograd.Function):
        """
        This class wraps the KIMModel for compatibility with torch autograd.
        It uses numdifftools for differentiation over model parameters. This method would
        not accept dataset as a parameter
        """
        @staticmethod
        def forward(ctx, model, configuration, parameters:torch.Tensor):
            """
            Forward function for computing model with a given configuration and parameters
            :param ctx: Context object of the autograd function
            :param model: :class:`kliff.models.KIMModel`
            :param configuration: :class:`kliff.dataset.Configuration`
            :param parameters: Parameter tensor to accumulate gradients in the backward pass (`parameters.grad`)
            :return: Model tensor
            """
            model_param_fn = model.get_parameter_function(configuration)
            ctx.model_param_fn = model_param_fn
            ctx.parameters = parameters
            ctx.grad_fn = nd.Gradient(model_param_fn)
            return torch.tensor(model_param_fn(parameters.numpy()), requires_grad=True)

        @staticmethod
        def backward(ctx, grad_outputs):
            """
            Backward function for computing the gradients w.r.t parameters.
            Args:
            :param ctx: Context object of the autograd function
            :param grad_outputs: Input tensors for computing vector-Jacobian product
            :return: Gradients w.r.t parameters
            """
            grad_fn = ctx.grad_fn
            parameters = ctx.parameters
            dE_dparam = grad_fn(parameters.numpy())
            if type(dE_dparam) == np.float64 or dE_dparam.shape == ():
                dE_dparam = np.array([dE_dparam])
            input_grad = grad_outputs.numpy()
            dE_dparam = np.dot(dE_dparam, input_grad)
            dE_dparam = torch.from_numpy(dE_dparam)
            dE_dparam.requires_grad_(True)
            return None, None, dE_dparam


class TorchGraphFunction(torch.autograd.Function):
    """
    This class is a wrapper for enabling :class:`torch.autograd.Function` like API for the KIMTorchGraph class.
    It takes in a :class:`kliff.models.KIMTorchGraphGenerator` object as GraphCtx and generates a graph in
    forward pass and returns the graph tensors as outputs. The backward pass sums the gradients over the
    graph indices. This provides a smooth interface for differentiating the graph based models.
    """
    @staticmethod
    def forward(ctx, GraphCtx, configuration, coordinate:torch.Tensor):
        """
        Forward function for generating the graph and returning the graph tensors.
        :param ctx: Context object of the autograd function
        :param GraphCtx: :class:`kliff.models.KIMTorchGraphGenerator` instance
        :param configuration: :class:`kliff.dataset.Configuration` to compute the graph
        :param coordinate: Coordinate tensor to accumulate gradients in the backward pass (`coordinate.grad`)
        :return: Graph tuple
        """
        ctx.GraphCtx = GraphCtx
        graph = GraphCtx.generate_graph(configuration)
        outputs = [graph.species, graph.coords]
        for i in range(graph.n_layers):
            outputs.append(graph.__getattr__(f"edge_index{i}"))
        outputs.append(graph.contributions)
        ctx.graph = graph
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward function for computing the gradients w.r.t positions.
        Args:
        :param ctx: Context object of the autograd function
        :param grad_outputs: Input tensors for computing vector-Jacobian product
        :return: Gradients w.r.t positions
        """
        graph = ctx.graph
        images = graph.images
        d_coords = grad_outputs[1]
        d_coords = scatter(d_coords, images, 0)
        return None, None, d_coords

