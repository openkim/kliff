import warnings
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union

import numpy as np
import pickle


class Parameter(np.ndarray):
    """
    A Parameter class that inherits from numpy.ndarray. This class is used to store parameters for optimization. It is a numpy array with additional attributes such as name, transform, etc.
    Parameters
    ----------
    input_array : array_like
        Input array to be converted to Parameter object.
    name : str, optional
        Name of the parameter.
    transform : Transform, optional
        Transform object to be applied to the parameter.
    bounds : tuple, optional
        Bounds for the parameter. If None, no bounds are applied.
    is_trainable : bool, optional
        If True, the parameter is trainable. If False, the parameter is not trainable.
    index : int, optional
        Index of the parameter in the parameter vector.
    opt_mask : array_like, optional
        A boolean array of the same shape as the parameter. If True, the parameter is optimized. If False, the parameter is not optimized.

    Methods
    -------
    transform_() : None
        Apply the transform to the parameter.
    inverse_transform_() : None
        Apply the inverse transform to the parameter.
    reset_() : None
        Reset the parameter to its original value.
    get_transformed_array() : array_like
        Get the transformed array.
    copy_to_param_(arr) : None
        Copy arr to self. arr can be a numpy array or a Parameter object. This method assumes that the array is of the same type and shape as self,
        compensated for opt_mask. If not, it will raise an error.
    copy_to_param_transformed_(arr) : None
        Copy arr to self. arr can be a numpy array or a Parameter object. This method enforces transformation on the array and parameter before copying.
    get_numpy_array() : array_like
        Get a numpy array of parameters in the original space. this does not apply any opt_mask. This ensures the correctness of the array for optimization/other applications.
    get_numpy_opt_array() : array_like
        Get a numpy array of parameters in the original space. This applies opt_mask. This ensures the correctness of the array for optimization/other applications.

    """

    def __new__(cls, input_array, name=None, transform=None, bounds=None, is_trainable=False,index=None,opt_mask=None):
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        obj.transform = transform
        obj.original = input_array
        obj.index = index
        obj.is_transformed = False
        obj.bounds = bounds
        # if isinstance(clamp, float) or isinstance(clamp, int):
        #     obj.clamp = (clamp, clamp)
        # else:
        #     obj.clamp = clamp
        obj.is_trainable = is_trainable
        if opt_mask:
            obj.opt_mask = opt_mask
        else:
            obj.opt_mask = np.ones_like(obj,dtype=bool)
        obj.bounds_transformed = False
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', None)
        self.transform = getattr(obj, 'transform_fn', None)
        self.original = getattr(obj, 'original', None)
        # self.clamp = getattr(obj, 'clamp', None)
        self.bounds = getattr(obj, 'bounds', None)
        self.is_trainable = getattr(obj, 'is_trainable', False)
        self.index = getattr(obj, 'index', None)
        self.is_transformed = getattr(obj, 'is_transformed', False)
        self.opt_mask = getattr(obj, 'opt_mask', None)
        self.bounds_transformed = getattr(obj, 'bounds_transformed', False)

    def __repr__(self):
        return "Parameter {0}:".format(self.name) + np.ndarray.__repr__(self)

    def transform_(self):
        if self.is_transformed:
            warnings.warn("Parameter {0} has already been transformed.".format(self.name))
            return
        else:
            self.is_transformed = True
            if self.transform is not None:
                for i in range(len(self)):
                    self[i] = self.transform(self[i])

    def inverse_transform_(self):
        if not self.is_transformed:
            warnings.warn("Parameter {0} has not been transformed.".format(self.name))
            return
        else:
            if self.transform is not None:
                for i in range(len(self)):
                    self[i] = self.transform.inverse(self[i])
            self.is_transformed = False

    def reset_(self):
        self[:] = self.original

    def get_transformed_array(self):
        return self.transform(self)

    def copy_to_param_(self, arr):
        """
        Copy arr to self. arr can be a numpy array or a Parameter object. This method assumes that the array is of the same type and shape as self,
        compensated for opt_mask. If not, it will raise an error.
        """
        try:
            if self.opt_mask is not None:
                tmp_arr = np.zeros_like(self)
                tmp_arr[self.opt_mask] = arr
                tmp_arr[~self.opt_mask] = self[~self.opt_mask]
                # TODO: use copy_at_ instead of this?
                arr = tmp_arr
            arr = arr.astype(self.dtype)
        except AttributeError:
            arr = np.array(arr).astype(self.dtype)
        self[:] = arr

    def copy_to_param_transformed_(self, arr):
        """
        Copy arr to self. arr can be a numpy array or a Parameter object. This method enforces transformation on the array and parameter before copying.
        """
        # transform the array and ensure that the parameter is transformed
        arr = self.transform(arr)
        if not self.is_transformed:
            self.transform_()
        self.copy_to_param_(arr)

    def get_numpy_array(self):
        """
        Get a numpy array of parameters in the original space. this does not apply any opt_mask. This ensures the correctness of the array for optimization/other applications.
        """
        if (self.transform is not None) and self.is_transformed:
            return self.transform.inverse(self)
        else:
            return self

    def get_numpy_opt_array(self):
        """
        Get a numpy array of parameters in the original space. this does apply the opt_mask. This ensures the correctness of the array for optimization/other applications.
        """
        np_arr = self.get_numpy_array()
        if self.opt_mask is not None:
            np_arr = np_arr[self.opt_mask]
        return np_arr

    def copy_at_(self, arr, index):
        """
        same as copy_ but for selected indices.
        """
        if isinstance(index, int):
            index = [index]
            arr = np.array([arr])
        arr = arr.astype(self.dtype)
        for i, j in zip(index, arr):
            self[i] = j

    def save(self, filename):
        data_dict = self.__dict__.copy()
        # save pkl
        with open(filename, "wb") as f:
            pickle.dump(data_dict, f)

    def load(self, filename):
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
        self.is_transformed = data_dict["is_transformed"]
        self.opt_mask = data_dict["opt_mask"]
        # self.clamp_()
        return self

    def add_transform_(self, transform):
        self.transform = transform
        self.transform_()
        self.is_transformed = True
        if self.bounds is not None and not self.bounds_transformed:
            self.bounds = self.transform(self.bounds)

    def add_bounds(self, bounds):
        """
        Add bounds to the parameter. Must be in original space. The bounds will be transformed if the parameter is transformed.
        :param bounds:
        :return:
        """
        if bounds.shape[1] != 2:
            raise ValueError("Bounds must have shape (n, 2).")
        if self.transform is not None:
            self.bounds = self.transform(bounds)
            self.bounds_transformed = True
        else:
            self.bounds = bounds

    def add_transformed_bounds(self, bounds):
        """
        Add bounds to the parameter. Must be in transformed space. Does not do any additional checks.
        :param bounds:
        :return:
        """
        if bounds.shape[1] != 2:
            raise ValueError("Bounds must have shape (n, 2).")
        self.bounds = bounds
        self.bounds_transformed = True

    def add_opt_mask(self, mask):
        if mask.shape != self.shape:
            raise ValueError("Mask must have shape {0}.".format(self.shape))
        self.opt_mask = mask

    def get_formatted_param_bounds(self):
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
        return self.bounds is not None

    def get_inverse_bounds(self):
        return self.transform.inverse(self.bounds)

# class _Index:
#     """
#     Mapping of a component of the optimizing parameter list to the model parameter dict.
#     """
#
#     def __init__(self, name, parameter_index=None, component_index=None):
#         self.name = name
#         self.parameter_index = self.p_idx = parameter_index
#         self.component_index = self.c_idx = component_index
#
#     def set_parameter_index(self, index):
#         self.parameter_index = self.p_idx = index
#
#     def set_component_index(self, index):
#         self.component_index = self.c_idx = index
#
#     def __expr__(self):
#         return self.name
#
#     def __eq__(self, other):
#         if isinstance(other, self.__class__):
#             return self.__dict__ == other.__dict__
#         else:
#             return False
#
#     def __ne__(self, other):
#         if isinstance(other, self.__class__):
#             return self.__dict__ != other.__dict__
#         else:
#             return True
#
#
# def _remove_comments(lines: List[str]):
#     """
#     Remove lines in a string list that start with # and content after #.
#     """
#     processed_lines = []
#     for line in lines:
#         line = line.strip()
#         if not line or line[0] == "#":
#             continue
#         if "#" in line:
#             line = line[0 : line.index("#")]
#         processed_lines.append(line)
#     return processed_lines
#
#
# def _check_shape(x: Any, key="parameter"):
#     """Check x to be a 1D array or list-like sequence."""
#     if isinstance(x, np.ndarray):
#         x = x.tolist()
#     if isinstance(x, Sequence):
#         if any(isinstance(i, Sequence) for i in x):
#             raise ParameterError(f"{key} should be a 1D array (or list).")
#     else:
#         raise ParameterError(f"{key} should be a 1D array (or list).")
#
#     return x.copy()


class ParameterError(Exception):
    def __init__(self, msg):
        super(ParameterError, self).__init__(msg)
        self.msg = msg
