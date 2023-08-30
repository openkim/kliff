import warnings
# from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union
from kliff.transforms import ParameterTransform
import numpy as np
import pickle


# This file uses MyST format
class Parameter(np.ndarray):
    """Parameter class for containing physics-based model parameters.

    Modeled on `torch.nn.Parameters`, it inherits from `numpy.ndarray`. It is a numpy array with additional attributes such as name, transform, etc. Functions with
    "_" suffix modifies the value that the parameter holds.
    For maintaining compatibility, use `get_numpy_array` (for
    getting a numpy array of parameters) and `get_numpy_opt_array`(transformed numpy array but with only the optimizable
    values).

    Attributes:
    name : Name of the parameter.
    transform : Instance of  ``ParameterTransform`` object to be applied to the parameter.
    index : Index of the parameter in the parameter vector. used for setting the parameter in the KIMPY.
    bounds : Bounds for the parameter, must be numpy array of shape n x 2, with [n,0] as lower bound, and [n,1] as the upper bound. If None, no bounds are applied.
    is_trainable : If True, the parameter is trainable, and will be passed to scipy optimizer. If False, the parameter is not trainable, and kept constant.
    opt_mask : A boolean array of the same shape as the parameter. If True, the parameter is optimized. If False, the parameter is not optimized.

    """

    def __new__(cls, input_array: np.ndarray, name: str = None, transform:ParameterTransform = None, bounds:np.ndarray = None, is_trainable:bool = False, index:int = None, opt_mask:np.ndarray = None):
        """Initializes and returns a new instance of Parameter.

        Args:
            input_array: Input numpy array to initialize the parameter with.
            name: Name of the parameter
            transform: Instance of  ``ParameterTransform`` object to be applied to the parameter.
            bounds: n x 2 array of lower and upper bounds for the parameter. If None, no bounds are applied
            is_trainable: boolean if the parameter is trainable, and will be passed to scipy optimizer.
            index: Index of the parameter in the parameter vector. Used for setting the parameter in the KIMPY.
            opt_mask: Boolean array of the same shape as the parameter. The values marked ``True`` are optimized, and ``False`` are not optimized.

        Returns:
            A new instance of Parameter.
        """
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        obj.transform = transform
        obj._original = input_array
        obj.index = index
        obj._is_transformed = False
        obj.bounds = bounds
        obj.is_trainable = is_trainable
        if opt_mask:
            obj.opt_mask = opt_mask
        else:
            obj.opt_mask = np.ones_like(obj,dtype=bool)
        obj._bounds_transformed = False
        return obj

    def __array_finalize__(self, obj):
        """Finalizes a parameter, needed for numpy object cleanup."""
        if obj is None: return
        self.name = getattr(obj, 'name', None)
        self.transform = getattr(obj, 'transform_fn', None)
        self.original = getattr(obj, 'original', None)
        self.bounds = getattr(obj, 'bounds', None)
        self.is_trainable = getattr(obj, 'is_trainable', False)
        self.index = getattr(obj, 'index', None)
        self._is_transformed = getattr(obj, '_is_transformed', False)
        self.opt_mask = getattr(obj, 'opt_mask', None)
        self._bounds_transformed = getattr(obj, '_bounds_transformed', False)

    def __repr__(self):
        return "Parameter {0}:".format(self.name) + np.ndarray.__repr__(self)

    def transform_(self):
        """Apply the transform to the parameter.

        This method simple applies the function ``ParameterTransform.__call__`` to the parameter
        (or equivalently, ``ParameterTransform.transform()``)."""
        if self._is_transformed:
            # warnings.warn("Parameter {0} has already been transformed.".format(self.name))
            # Warnings become quite noisy, so commenting it out for now.
            # TODO: figure out a better solution for this.
            return
        else:
            if self.transform is not None:
                for i in range(len(self)):
                    self[i] = self.transform(self[i])
            self._is_transformed = True

    def inverse_transform_(self):
        """Apply the inverse transform to the parameter, simply applies the function ``ParameterTransform.inverse()`` to the parameters."""
        if not self._is_transformed:
            warnings.warn("Parameter {0} has not been transformed.".format(self.name))
            return
        else:
            if self.transform is not None:
                for i in range(len(self)):
                    self[i] = self.transform.inverse(self[i])
            self._is_transformed = False # Raises style warning, but is lot simpler and cleaner.

    def reset_(self):
        """Reset the parameter to its original value."""
        self[:] = self._original

    def get_transformed_array(self):
        """Applies the transform to the parameter, and returns the transformed array."""
        return self.transform(self)

    def copy_to_param_(self, arr):
        """Copy array to self in the original space.

        Array can be a numpy array or a Parameter object.
        This method assumes that the array is of the same type and shape as self,
        compensated for opt_mask. If not, it will raise an error. This method assumes that the incoming array
        is in the original space.

        Args:
            arr: Array to copy to self.
        """
        try:
            if self.opt_mask is not None:
                tmp_arr = np.zeros_like(self)
                tmp_arr[self.opt_mask] = arr
                tmp_arr[~self.opt_mask] = self[~self.opt_mask]
                arr = tmp_arr
            arr = arr.astype(self.dtype)
        except AttributeError:
            arr = np.array(arr).astype(self.dtype)
        self[:] = arr

    def copy_to_param_transformed_(self, arr):
        """Copy arr to transformed self.

        Array can be a numpy array or a Parameter object. This method assumes that the incoming array is in the transformed space.
        If the Parameter is not transformed, this method will transform it first.

        Args:
            arr: Array to copy to self.
        """
        # transform the array and ensure that the parameter is transformed
        arr = self.transform(arr)
        if not self._is_transformed:
            self.transform_()
        self.copy_to_param_(arr)

    def get_numpy_array(self):
        """ Get a numpy array of parameters in the original space.

        This method should be uses for getting the numpy array of parameters where the ``Parameters`` class might not work.
        Biggest example of it is passing to the optimizer as the optimizer might overwrite or destroy the parameters.

        Returns:
            A numpy array of parameters in the original space.
        """
        if (self.transform is not None) and self._is_transformed:
            return self.transform.inverse(self)
        else:
            return self

    def get_numpy_opt_array(self):
        """Get a masked numpy array of parameters in the original space.

        This method is same as ``get_numpy_array`` but additionally does apply the opt_mask. This ensures the correctness
        of the array for optimization/other applications. This should be the defacto method for getting the numpy array
        of parameters.

        Returns:
            A numpy array of parameters in the original space.
        """
        np_arr = self.get_numpy_array()
        if self.opt_mask is not None:
            np_arr = np_arr[self.opt_mask]
        return np_arr

    def copy_at_(self, arr, index):
        """Copy raw values at a particular index or indices."""
        if isinstance(index, int):
            index = [index]
            arr = np.array([arr])
        arr = arr.astype(self.dtype)
        for i, j in zip(index, arr):
            self[i] = j

    def save(self, filename):
        """Saves the parameter as a pickled dict to a file."""
        data_dict = self.__dict__.copy()
        # save pkl
        with open(filename, "wb") as f:
            pickle.dump(data_dict, f)

    def load(self, filename):
        """Loads the parameter from a pickled dict to a file."""
        # load dict from pkl and assign
        with open(filename, "rb") as f:
            data_dict = pickle.load(f)
        self.name = data_dict["name"]
        self.transform = data_dict["transform"]
        self.original = data_dict["original"]
        self.clamp = data_dict["clamp"]
        self.is_trainable = data_dict["is_trainable"]
        self[:] = data_dict["data"]
        self.index = data_dict["index"]
        self._is_transformed = data_dict["_is_transformed"]
        self.opt_mask = data_dict["opt_mask"]
        # self.clamp_()
        return self

    def add_transform_(self, transform: ParameterTransform):
        """Save a transform object with the parameter."""
        self.transform = transform
        self.transform_()
        self._is_transformed = True
        if self.bounds is not None and not self._bounds_transformed:
            self.bounds = self.transform(self.bounds)

    def add_bounds(self, bounds:np.ndarray):
        """Add bounds to the parameter.
        Must be in original space. The bounds will be transformed if the parameter is transformed.
        Args:
            bounds: numpy array of shape (n, 2)
        """
        if bounds.shape[1] != 2:
            raise ValueError("Bounds must have shape (n, 2).")
        if self.transform is not None:
            self.bounds = self.transform(bounds)
            self._bounds_transformed = True
        else:
            self.bounds = bounds

    def add_transformed_bounds(self, bounds: np.ndarray):
        """
        Add bounds to the parameter. Must be in transformed space. Does not do any additional checks.
        Args:
            bounds: numpy array of shape (n, 2)
        """
        if bounds.shape[1] != 2:
            raise ValueError("Bounds must have shape (n, 2).")
        self.bounds = bounds
        self._bounds_transformed = True

    def add_opt_mask(self, mask: np.ndarray):
        """Set mask for optimizing vector quantities.

        It expects an input array of shape (n,), where n is the dimension of the vector quantity to be optimized.
        This array must contain n booleans indicating which properties to optimize.

        Args:
            mask: boolean array of same shape as the vector quantity to be optimized
        """
        if mask.shape != self.shape:
            raise ValueError("Mask must have shape {0}.".format(self.shape))
        self.opt_mask = mask

    def get_formatted_param_bounds(self):
        """Returns bounds array that is used by scipy optimizer.
        """
        arr = self.get_numpy_opt_array()
        bounds = []
        if self.bounds is not None:
            if (self.bounds.shape[0] == arr.shape[0]) and (self.bounds.shape[1] == 2):
                for i in range(arr.shape[0]):
                    bounds.append((self.bounds[i, 0], self.bounds[i, 1]))
            else:
                raise ValueError("Bounds must have shape: {0}x2.".format(arr.shape))
        else:
            bounds = [(None, None) for i in range(arr.shape[0])]
        return bounds

    def has_opt_params_bounds(self):
        """Check if bounds are set for optimizing quantities
        """
        return self.bounds is not None

    def get_inverse_bounds(self):
        return self.transform.inverse(self.bounds)


class ParameterError(Exception):
    def __init__(self, msg):
        super(ParameterError, self).__init__(msg)
        self.msg = msg
