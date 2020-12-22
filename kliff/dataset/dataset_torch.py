import torch
from torch.utils.data import Dataset

from ..descriptors.descriptor import load_fingerprints


class FingerprintsDataset(Dataset):
    r"""Atomic environment fingerprints dataset.

    Parameters
    ----------
    path: string
        Path to the fingerprints file.

    transform: callable (optional)
        Optional transform to be applied on a sample.
    """

    def __init__(self, path, transform=None):
        self.fp = load_fingerprints(path)
        self.transform = transform

    def __len__(self):
        return len(self.fp)

    def __getitem__(self, index):
        sample = self.fp[index]
        if self.transform:
            sample = self.transform(sample)
        return sample


def fingerprints_collate_fn(batch):
    r"""Convert a batch of samples into tensor.

    Unlike the default_collate_fn(), which stack samples in the batch (requiring each
    sample having the same dimension), this function does not do the stack.

    Parameters
    ----------
    batch: list
        A batch of samples.

    Returns
    -------
    tensor_batch: list
        Transform each sample into a tensor.
    """
    tensor_batch = []
    for i, sample in enumerate(batch):
        tensor_sample = {}
        for key, value in sample.items():
            if type(value).__module__ == "numpy":
                value = torch.from_numpy(value)
            tensor_sample[key] = value
        tensor_batch.append(tensor_sample)

    return tensor_batch
