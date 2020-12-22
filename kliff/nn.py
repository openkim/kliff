import torch
from torch.nn import *


# redefine Dropout layer
class Dropout(torch.nn.modules.dropout._DropoutNd):
    r"""A Dropout layer that zeros the same element of descriptor values for all
    atoms.

    Note `torch.nn.Dropout` dropout each component independently.

    Parameters
    ----------
    p: float
        probability of an element to be zeroed. Default: 0.5

    inplace: bool
        If set to `True`, will do this operation in-place. Default: `False`

    Shapes
    ------
        Input: [N, D] or [1, N, D]
        Output: [N, D] or [1, N, D] (same as Input)
        The first dimension 1 is because the dataloader provides only sample each
        iteration.
    """

    def forward(self, input):
        dim = input.dim()
        shape = input.shape
        if dim == 2:
            shape_4D = (1, *shape, 1)
        elif dim == 3:
            if shape[0] != 1:
                raise Exception("Shape[0] needs to be 1 for a 3D tensor.")
            shape_4D = (*shape, 1)

        else:
            raise Exception(
                "Input need to be 2D or 3D tensor, but got a " "{}D tensor.".format(dim)
            )
        x = torch.reshape(input, shape_4D)
        x = torch.transpose(x, 1, 2)
        y = torch.nn.functional.dropout2d(x, self.p, self.training, self.inplace)
        y = torch.transpose(y, 1, 2)
        y = torch.reshape(y, shape)
        return y
