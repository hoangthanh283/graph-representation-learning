from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import torch
from tabulate import tabulate
from torch_geometric.datasets import Planetoid

from gnn import PRJ_PATH


class PlanetoidDatasetName(Enum):
    CORA = "Cora"
    CITE_SEER = "CiteSeer"
    PUBMED = "PubMed"


class Split(Enum):
    PUBLIC = "public"
    COMPLETE = "complete"
    FULL = "full"


class DatapileAdaptedPlanetoid(NamedTuple):
    """
    Instance to store the planetoid data which was converted to Datapile format.
    """
    x: torch.Tensor  # Feature matrix with size of 1 x N x H.
    y: torch.Tensor  # Node labels with size of 1 x N.
    adj_matrix: torch.Tensor  # Binary adjacency matrix with size 1 x N x 1 x N with adj_matrix[0, i, 0, j] = 1 means
    # that the node ith links to not jth.
    train_mask: torch.Tensor  # Training mask with size of N.
    val_mask: torch.Tensor  # Validation mask with size of N.
    test_mask: torch.Tensor  # Testing mask with size of N.
    num_classes: int  # Number of classes.

    def __str__(self, indent: int = 4):
        table_data = [[key, value.shape] for key, value in self._asdict().items()]
        table = tabulate(table_data,
                         headers=[],
                         tablefmt='simple')
        return '\n'.join(' ' * indent + line for line in table.splitlines())

    def get_input_dim(self) -> int:
        return self.x.shape[-1]



def get_adj_matrix(edge_index: torch.Tensor, num_nodes: int) -> torch.FloatTensor:
    """
    Construct dense ADJ matrix from edge indexes.
    """
    # Convert edge_indexes to scipy sparse matrix.
    adj_matrix = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0].numpy(), edge_index[1].numpy())),
                               shape=(num_nodes, num_nodes), dtype=np.float32)
    adj_matrix.setdiag(1)
    return torch.FloatTensor(adj_matrix.todense())


def get_planetoid_dataset(dataset_name: PlanetoidDatasetName, normalize_features=False,
                          split=Split.PUBLIC) -> DatapileAdaptedPlanetoid:
    """
    Get the planetoid dataset which is capable for loading ['Cora', 'CiteSeer', 'PubMed'] datasets. The original code is
    from "https://github.com/russellizadi/ssp/blob/master/experiments/datasets.py".
    :param dataset_name: Dataset name
    :param normalize_features: Whether to apply normalization.
    :param split: Whether to normalize the dataset.
    :return: Planetoid dataset.
    """
    path_to_save = Path.joinpath(PRJ_PATH, dataset_name.value)
    if split == Split.COMPLETE:
        dataset = Planetoid(path_to_save, dataset_name.value)

        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[:dataset[0].num_nodes - 1000] = 1

        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1

        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500:] = 1
    else:
        dataset = Planetoid(path_to_save, dataset_name.value, split=split.value)

    x = dataset[0].x
    if normalize_features:
        x = torch.nn.functional.normalize(x)
    y = dataset[0].y
    train_mask = dataset[0].train_mask
    val_mask = dataset[0].val_mask
    test_mask = dataset[0].test_mask
    adj = get_adj_matrix(dataset[0].edge_index, dataset[0].num_nodes)
    num_classes = dataset.num_classes

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        adj = adj.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    return DatapileAdaptedPlanetoid(x=x, y=y, adj_matrix=adj, train_mask=train_mask, val_mask=val_mask,
                                    test_mask=test_mask, num_classes=num_classes)
