import torch
from torch import nn
from torch_geometric import nn as gnn

# Misc
from torch_scatter import scatter
from typing import Tuple

# Equivarient GNN, Saritorious et al. 2020
# ------------------ KIM Compatible Graph Convolution Layer ------------------ #
class EGCL4KLIFF(gnn.MessagePassing):

    propagate_type = {
        "h": torch.Tensor,
        "r": torch.Tensor
    }

    def __init__(self,
                 in_node_fl,
                 hidden_node_fl,
                 edge_fl=0,
                 act_fn=nn.SiLU()):
        super().__init__(aggr="add")
        self.in_node_fl = int(in_node_fl)
        self.hidden_node_fl = int(hidden_node_fl)
        self.edge_fl = int(edge_fl)

        self.normalize_radial = False
        self.C = torch.zeros(1)

        self.phi_e = nn.Sequential(
            nn.Linear(self.hidden_node_fl * 2 + 1,
                      self.hidden_node_fl), act_fn,
            nn.Linear(self.hidden_node_fl, self.hidden_node_fl), act_fn)

        self.phi_r = nn.Sequential(
            nn.Linear(self.hidden_node_fl, hidden_node_fl), act_fn,
            nn.Linear(self.hidden_node_fl, 1), nn.Tanh())

        self.phi_h = nn.Sequential(
            nn.Linear(self.hidden_node_fl + self.hidden_node_fl,
                      self.hidden_node_fl), act_fn,
            nn.Linear(self.hidden_node_fl, self.hidden_node_fl))

    def forward(self, h: torch.Tensor, r: torch.Tensor,
                edge_index: torch.Tensor):
        n_atoms = h.size(0)
        self.C = torch.tensor(1. / (n_atoms - 1.))
        h1, r = self.propagate(edge_index, h=h, r=r, size=None)
        mask = h1.abs().sum(1)
        mask[mask != 0] = -1.
        mask = mask + 1
        h1 = h1 + mask.reshape(-1, 1) * h
        return h1, r

    def message(self, h_i, h_j, r_i, r_j,
                h) -> Tuple[torch.Tensor, torch.Tensor]:
        dr = r_j - r_i
        norm_dr = dr.pow(2).sum(1)
        if self.normalize_radial:
            dr = dr / norm_dr
        mij = self.phi_e(torch.cat([h_i, h_j, torch.unsqueeze(norm_dr, 1)], 1))
        # pad_length = mij.size(0) - h.size(0) if (
        #     mij.size(0) > h.size(0)) else 0
        # mij = torch.nn.functional.pad(mij.T, (0, pad_length)).T
        delrij = dr * self.phi_r(mij)
        return mij, delrij

    def aggregate(self, messages: Tuple[torch.Tensor, torch.Tensor], index,
                  r) -> Tuple[torch.Tensor, torch.Tensor]:
        m = scatter(messages[0], index, dim=0, reduce="add")
        dr = scatter(messages[1], index, dim=0, reduce="add")
        pad_length = r.size(0) - dr.size(0) if (r.size(0) > dr.size(0)) else 0
        r_ = r + torch.nn.functional.pad(dr.T, (0, pad_length)).T
        m = torch.nn.functional.pad(m.T, (0, pad_length)).T
        return m, r_

    def update(self, messages: Tuple[torch.Tensor, torch.Tensor],
               h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prime = self.phi_h(torch.cat([h, messages[0]], 1))
        mask = torch.sum(torch.abs(messages[0]), 1)
        mask[mask > 0] = 1.
        mask = torch.reshape(mask, (-1, 1))
        h = h * torch.abs(mask - 1.) + h_prime * mask
        return h, messages[1]

# ------------------ KIM Compatible Graph Neural Network ------------------ #
class EGNN4KLIFF(nn.Module):

    def __init__(self, in_node_fl, hidden_node_fl, n_conv_l, act_fn=nn.SiLU()):
        super(EGNN4KLIFF, self).__init__()
        self.in_node_fl = in_node_fl
        self.hidden_node_fl = hidden_node_fl
        self.n_conv_l = n_conv_l

        self.conv_module = nn.ModuleList([
            EGCL4KLIFF(hidden_node_fl, hidden_node_fl).jittable()
            for i in range(n_conv_l)
        ])

        self.mlp = nn.Sequential(nn.Linear(hidden_node_fl,
                                           hidden_node_fl), act_fn,
                                 nn.Linear(hidden_node_fl, hidden_node_fl),
                                 act_fn, nn.Linear(hidden_node_fl, 1))

        self.embedding = nn.Linear(self.in_node_fl, self.hidden_node_fl)
        self.register_buffer("pow_vec", torch.arange(3))

    def forward(self, x: torch.Tensor, r: torch.Tensor,
                edge_index0: torch.Tensor, edge_index1: torch.Tensor,
                edge_index2: torch.Tensor, contributions: torch.Tensor):
        h0 = torch.unsqueeze(x / torch.max(x), 1)
        h0 = h0.pow(self.pow_vec)
        h = self.embedding(h0.double())
        h, r = self.conv_module[0](h, r, edge_index2)
        h, r = self.conv_module[1](h, r, edge_index1)
        h, r = self.conv_module[2](h, r, edge_index0)
        E_local = self.mlp(h)
        E = scatter(E_local, contributions.long(), dim=0, reduce="add")
        return E[::2]


