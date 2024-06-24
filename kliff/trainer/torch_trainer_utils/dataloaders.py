from typing import Any, List, Tuple, Union

import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Dataset as TorchGeometricDataset

from kliff.dataset import Dataset
from kliff.transforms.configuration_transforms import (
    Descriptor,
    KIMDriverGraph,
    PyGGraph,
)


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
        device = config_0.device
        ptr = torch.tensor([0], dtype=torch.int64, device=device)

        # extract the fingerprint fields
        n_atoms_0 = torch.tensor(
            [config_0["n_atoms"]], dtype=torch.int64, device=device
        )
        species_0 = torch.tensor(config_0["species"], dtype=torch.int64, device=device)
        neigh_list_0 = torch.tensor(
            config_0["neigh_list"], dtype=torch.int64, device=device
        )
        num_neigh_0 = torch.tensor(
            config_0["num_neigh"], dtype=torch.int64, device=device
        )
        image_0 = torch.tensor(config_0["image"], dtype=torch.int64, device=device)
        coords_0 = torch.tensor(config_0["coords"], device=device)
        descriptors_0 = torch.tensor(config_0["descriptor"], device=device)
        contribution_0 = torch.zeros(
            descriptors_0.shape[0], dtype=torch.int64, device=device
        )
        batch_len = len(batch)
        ptr_shift = torch.tensor(coords_0.shape[0], dtype=torch.int64, device=device)

        for prop in self.consistent_properties:
            property_dict_0[prop] = torch.as_tensor(
                property_dict_0[prop], device=device
            )

        for i in range(1, batch_len):
            config_i, property_dict_i = batch[i]

            n_atoms_i = config_i["n_atoms"]
            species_i = config_i["species"]
            neigh_list_i = config_i["neigh_list"]
            num_neigh_i = config_i["num_neigh"]
            image_i = config_i["image"]
            coords_i = config_i["coords"]
            descriptors_i = config_i["descriptor"]
            contribution_i = (
                torch.zeros(descriptors_i.shape[0], dtype=torch.int64, device=device)
                + i
            )

            n_atoms_0 = torch.cat(
                (n_atoms_0, torch.tensor([n_atoms_i], dtype=torch.int64)), 0
            )
            species_0 = torch.cat(
                (species_0, torch.tensor(species_i, dtype=torch.int64)), 0
            )
            neigh_list_0 = torch.cat(
                (
                    neigh_list_0,
                    torch.tensor(neigh_list_i, dtype=torch.int64) + ptr_shift,
                ),
                0,
            )
            num_neigh_0 = torch.cat(
                (num_neigh_0, torch.tensor(num_neigh_i, dtype=torch.int64)), 0
            )
            image_0 = torch.cat(
                (image_0, torch.tensor(image_i, dtype=torch.int64) + ptr_shift), 0
            )
            coords_0 = torch.cat((coords_0, torch.tensor(coords_i)), 0)
            descriptors_0 = torch.cat((descriptors_0, torch.tensor(descriptors_i)), 0)
            contribution_0 = torch.cat(
                (contribution_0, torch.tensor(contribution_i)), 0
            )

            for prop in self.consistent_properties:
                property_dict_0[prop] = torch.vstack(
                    (property_dict_0[prop], torch.as_tensor(property_dict_i[prop]))
                )

            ptr = torch.cat((ptr, torch.tensor([ptr_shift], dtype=torch.int64)), 0)
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
        }

    def add_transform(self, transform: Descriptor):
        self.transform = transform


class GraphDataset(TorchGeometricDataset):
    """
    This class is a wrapper for the :class:`torch_geometric.data.Dataset` class to enable
    the use of :class:`kliff.dataset.Dataset` as a data source for the graph based models.
    """

    def __init__(self, dataset: Dataset, transform: KIMDriverGraph = None):
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
