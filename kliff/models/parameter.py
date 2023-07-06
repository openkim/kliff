import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union

import numpy as np
import pickle


class Parameter(np.ndarray):
    """
    Torch like parameters, but with numpy. Methods ending with _ modify the object in place.
    Trying to keep the terminology same but it seems a bit difficult. Subclassing ndarray.
    Make it immutable?
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
            # self.clamp_()

    def inverse_(self):
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

    def transformed(self):
        return self.transform(self)

    # def clamp_(self):
    #     if self.clamp is not None:
    #         self[:] = np.clip(self, self.clamp[0], self.clamp[1])

    def copy_raw_(self, inarray):
        self.original = inarray
        self[:] = self.transform(inarray)
        # self.clamp_()

    def copy_(self, arr):
        # if self.transform is not None:
        #     arr = self.transform(arr)
        try:
            arr = arr.astype(self.dtype)
        except AttributeError:
            arr = np.array(arr).astype(self.dtype)
        self[:] = arr
        # self.clamp_()

    def numpy(self):
        if (self.transform is not None) and self.is_transformed:
            return self.transform.inverse(self)
        else:
            return self

    def numpy_opt(self):
        np_arr = self.numpy()
        if self.opt_mask is not None:
            np_arr = np_arr[self.opt_mask]
        return np_arr

    def copy_at_(self, arr, index):
        arr = self.transform(arr)
        if isinstance(index, int):
            index = [index]
            arr = np.array([arr])
        arr = arr.astype(self.dtype)
        for i, j in zip(index, arr):
            self[i] = j
        # self.clamp_()

    def save(self, filename):
        data_dict = {
            "name": self.name,
            "transform": self.transform,
            "original": self.original,
            "clamp": self.clamp,
            "is_trainable": self.is_trainable,
            "data": self,
        }
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
        # self.clamp_()
        return self

    def add_transform_(self, transform):
        self.transform = transform

    def add_bounds_(self, bounds):
        """
        Add bounds to the parameter. Must be in transformed space
        :param bounds:
        :return:
        """
        if bounds.shape[1] != 2:
            raise ValueError("Bounds must have shape (n, 2).")
        self.bounds = bounds

    def finalize_(self):
        if self.is_transformed:
            warnings.warn("Parameter {0} has already been transformed.".format(self.name))
            return
        else:
            self.transform_()
            # self.clamp_()

    def set_opt_mask(self, mask):
        if mask.shape != self.shape:
            raise ValueError("Mask must have shape {0}.".format(self.shape))
        self.opt_mask = mask

    def get_opt_param_bounds(self):
        if self.bounds is not None:
            arr = self.numpy_opt()
            if ((self.opt_mask is not None) and
                (self.bounds.shape[0] == arr.shape) and
                (self.bounds.shape[1] == 2)) or \
                    (self.opt_mask is None):
                bounds = [(self.bounds[i, 0], self.bounds[i, 1]) for i in range(self.shape[0])]
                return bounds
            else:
                raise ValueError("Bounds must have shape: {0}x2.".format(arr.shape))

        else:
            return None

    def has_opt_params_bounds(self):
        return self.bounds is not None

    def get_inv_bounds(self):
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
