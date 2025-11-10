import sys

import torch
from torch.optim import Optimizer
from torch_geometric.data import DataLoader
from torch_geometric.nn import Node2Vec

from napistu_torch.napistu_data import NapistuData
from napistu_torch.utils.torch_utils import select_device


def get_node2vec_model(napistu_data: NapistuData, device: torch.device) -> Node2Vec:

    device = select_device(mps_valid=False)

    model = Node2Vec(
        napistu_data.edge_index,
        embedding_dim=128,
        walk_length=6,  # Short walks minimize hub convergence
        p=0.5,  # Discourage immediate backtracking
        q=2.0,  # Encourage exploring away from starting neighborhood
        walks_per_node=30,  # Many walks ensure coverage
        context_size=5,
        num_negative_samples=5,
        sparse=True,
    ).to(device)

    return model


def get_node2vec_training_regime(model: Node2Vec) -> tuple[DataLoader, Optimizer]:

    num_workers = 4 if sys.platform == "linux" else 0
    loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    return loader, optimizer


def get_node2vec_training_loop(
    model: Node2Vec, loader: DataLoader, optimizer: Optimizer, device: torch.device
) -> float:

    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)
