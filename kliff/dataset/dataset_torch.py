from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from kliff.legacy.descriptors.descriptor import load_fingerprints


class FingerprintsDataset(Dataset):
    """
    Atomic environment fingerprints dataset used by torch models.

    Args:
        filename: to the fingerprints file.
        transform: transform to be applied on a sample.
    """

    def __init__(self, filename: Path, transform: Optional[Callable] = None):
        self.fp = load_fingerprints(filename)
        for i, f in enumerate(self.fp):
            f["index"] = i

        self.transform = transform

    def __len__(self):
        return len(self.fp)

    def __getitem__(self, index):
        sample = self.fp[index]
        if self.transform:
            sample = self.transform(sample)
        return sample


def fingerprints_collate_fn(batch):
    """
    Convert a batch of samples into tensor.

    Unlike the default collate_fn(), which stack samples in the batch (requiring each
    sample having the same dimension), this function does not do the stack.

    Args:
        batch: A batch of samples.

    Returns:
        A list of tensor.
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
