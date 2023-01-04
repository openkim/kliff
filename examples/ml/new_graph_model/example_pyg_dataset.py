from kliff.ml.graphs import KIMTorchGraphGenerator
from kliff.dataset import Dataset

from torch.utils.data import DataLoader

# PyTorch Geometric Dependencies
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import InMemoryDataset

ds = Dataset(colabfit_database="colabfit_database", colabfit_dataset="my_si_dataset")

# Misc. examples
kgg = KIMTorchGraphGenerator(species=["Si"], cutoff=.77, n_layers=3, as_torch_geometric_data=True)

# PyTorch DataLoader
dl = DataLoader(ds, 5, collate_fn=kgg.collate_fn)


# %%
# ------------------ PyTorch Geometric ------------------ #
# Simple PyTorch Geometric dataset class
class SimplePYGDataset(InMemoryDataset):
    def __init__(self, data, transform):
        super().__init__("./", transform, None, None)
        self.data_list = data

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return [self.data_list[idx]]


# PyTorch Geometric Dataset
pyg_ds = SimplePYGDataset(ds, kgg.collate_fn)
# PyTorch Geometric DataLoader
pyg_dl = PyGDataLoader(pyg_ds, batch_size=5)

print("PyTorch Geometric DataLoader")
print(next(iter(pyg_dl)))

print("PyTorch DataLoader")
print(next(iter(dl)))
