from copy import deepcopy
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Dataset as TorchGeometricDataset

from kliff.dataset import Dataset
from kliff.transforms.configuration_transforms import Descriptor, PyGGraph, RadialGraph


class NeighborListDataset(TorchDataset):
    """
    TODO: format this dataset option
    batch should return a tuple of (species, config, neighbor_list, neighbors, contribution_index)
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx].fingerprint


class DescriptorDataset(TorchDataset):
    """
    This class is a wrapper for the :class:`torch.utils.data.Dataset` class to enable
    the use of :class:`kliff.dataset.Dataset` as a data source for the descriptor based models.
    It returns the fingerprint, properties and contribution index for each configuration.

    Args:
        dataset: :class:`kliff.dataset.Dataset` instance.
        property_keys: List of property keys to be used for the training. Default is
            ("energy", "forces", "stress"). These properties should be present on every
            configuration in the dataset. Among the provided list of properties, any
            property that is not present in any configuration will be ignored for all
            configurations.
    """

    def __init__(
        self,
        dataset: Dataset,
        property_keys: Union[Tuple, List] = ("energy", "forces", "stress"),
    ):
        self.dataset = dataset
        self.dataset.check_properties_consistency(property_keys)
        self.consistent_properties = self.dataset.get_metadata("consistent_properties")
        self.transform = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        property_dict = {
            key: getattr(self.dataset[idx], key) for key in self.consistent_properties
        }

        if self.transform:
            representation = self.transform(
                self.dataset[idx], return_extended_state=True
            )
        else:
            representation = self.dataset[idx].fingerprint
        return representation, property_dict

    def collate(self, batch: Any) -> dict:
        """
        Collate function for the dataset. This function takes in a batch of configurations
        and properties and returns the collated configuration, properties and contribution
        index tensors.
        Args:
            batch: list of configurations and properties for each configuration.

        Returns:
            dict: collated configuration, properties and contribution index tensors.
        """
        # get fingerprint and consistent properties
        config_0, property_dict_0 = batch[0]
        ptr = np.array([0], dtype=np.intc)

        # extract the fingerprint fields
        n_atoms_0 = np.array([config_0["n_atoms"]], dtype=np.intc)
        species_0 = np.array(config_0["species"], dtype=np.intc)
        neigh_list_0 = np.array(config_0["neigh_list"], dtype=np.intc)
        num_neigh_0 = np.array(config_0["num_neigh"], dtype=np.intc)
        image_0 = np.array(config_0["image"], dtype=np.intc)
        coords_0 = np.array(config_0["coords"])
        descriptors_0 = np.array(config_0["descriptor"])
        contribution_0 = np.zeros(descriptors_0.shape[0], dtype=np.intc)
        index_0 = np.array([config_0["index"]], dtype=np.intc)
        weight_0 = deepcopy(config_0["weight"])
        batch_len = len(batch)
        ptr_shift = np.array(coords_0.shape[0], dtype=np.intc)

        for prop in self.consistent_properties:
            property_dict_0[prop] = np.array(property_dict_0[prop])

        for i in range(1, batch_len):
            config_i, property_dict_i = batch[i]

            n_atoms_i = config_i["n_atoms"]
            species_i = config_i["species"]
            neigh_list_i = config_i["neigh_list"]
            num_neigh_i = config_i["num_neigh"]
            image_i = config_i["image"]
            coords_i = config_i["coords"]
            descriptors_i = config_i["descriptor"]
            weight_i = config_i["weight"]
            contribution_i = np.zeros(descriptors_i.shape[0], dtype=np.intc) + i

            n_atoms_0 = np.concatenate(
                (n_atoms_0, np.array([n_atoms_i], dtype=np.intc)), axis=0
            )
            species_0 = np.concatenate(
                (species_0, np.array(species_i, dtype=np.intc)), axis=0
            )
            neigh_list_0 = np.concatenate(
                (
                    neigh_list_0,
                    np.array(neigh_list_i, dtype=np.intc) + ptr_shift,
                ),
                axis=0,
            )
            num_neigh_0 = np.concatenate(
                (num_neigh_0, np.array(num_neigh_i, dtype=np.intc)), axis=0
            )
            image_0 = np.concatenate(
                (image_0, np.array(image_i, dtype=np.intc) + ptr_shift), axis=0
            )
            coords_0 = np.concatenate((coords_0, np.array(coords_i)), axis=0)
            descriptors_0 = np.concatenate(
                (descriptors_0, np.array(descriptors_i)), axis=0
            )
            contribution_0 = np.concatenate(
                (contribution_0, np.array(contribution_i)), axis=0
            )
            index_0 = np.concatenate(
                (index_0, np.array([config_i["index"]], dtype=np.intc)), axis=0
            )

            for key in weight_0:
                try:
                    weight_0[key] = np.concatenate(
                        (weight_0[key], np.array(weight_i[key])), axis=0
                    )
                except ValueError:
                    weight_0[key] = np.concatenate(
                        (np.atleast_2d(weight_0[key]), np.atleast_2d(weight_i[key])),
                        axis=0,
                    )

            for prop in self.consistent_properties:
                property_dict_0[prop] = np.vstack(
                    (property_dict_0[prop], np.array(property_dict_i[prop]))
                )

            ptr = np.concatenate((ptr, np.array([ptr_shift], dtype=np.intc)), axis=0)
            ptr_shift += coords_i.shape[0]

        return {
            "n_atoms": n_atoms_0,
            "n_particle": n_atoms_0,
            "species": species_0,
            "neigh_list": neigh_list_0,
            "num_neigh": num_neigh_0,
            "image": image_0,
            "coords": coords_0,
            "descriptors": descriptors_0,
            "property_dict": property_dict_0,
            "ptr": ptr,
            "contribution": contribution_0,
            "index": index_0,
            "weight": weight_0,
        }

    def add_transform(self, transform: Descriptor):
        self.transform = transform


class GraphDataset(TorchGeometricDataset):
    """
    This class is a wrapper for the :class:`torch_geometric.data.Dataset` class to enable
    the use of :class:`kliff.dataset.Dataset` as a data source for the graph based models.
    """

    def __init__(self, dataset: Dataset, transform: RadialGraph = None):
        super().__init__("./", transform, None, None)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def len(self):
        return len(self.dataset)

    def get(self, idx: int):
        if self.transform is None:
            return PyGGraph.from_dict(self.dataset[idx].fingerprint)
        else:
            return self.dataset[idx]
