import torch
from torch.utils.data import Dataset, DataLoader
from ..descriptors.descriptor import load_fingerprints


class FingerprintsDataset(Dataset):
    """Atomic environment fingerprints dataset.

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


class FingerprintsDataLoader(DataLoader):
    """A dataset loader that incorporate the support the number of epochs.

    The dataset loader will load an element from the next batch if a batch is
    fully iterated. This, in effect, looks like concatenating the dataset the
    number of epochs times.

    Parameters
    ----------
    num_epochs: int
        Number of epochs to iterate through the dataset.
    """

    def __init__(self, num_epochs=1, *args, **kwargs):
        super(FingerprintsDataLoader, self).__init__(*args, **kwargs)
        self.num_epochs = num_epochs
        self.epoch = 0
        self.iterable = None

    def next_element(self):
        """ Get the next data element.
        """
        if self.iterable is None:
            self.iterable = self._make_iterable()
        try:
            element = self.iterable.next()
        except StopIteration:
            self.epoch += 1
            if self.epoch == self.num_epochs:
                raise StopIteration
            else:
                self.iterable = self._make_iterable()
                element = self.next_element()
        return element

    def _make_iterable(self):
        iterable = iter(self)
        return iterable


def fingerprints_collate_fn(batch):
    """Convert a batch of samples into tensor.

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
            if type(value).__module__ == 'numpy':
                value = torch.from_numpy(value)
            tensor_sample[key] = value
        tensor_batch.append(tensor_sample)

    return tensor_batch
