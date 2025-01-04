from enum import Enum
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from gnn import PRJ_PATH


class PlanetoidDatasetName(Enum):
    CORA = "Cora"
    CITE_SEER = "CiteSeer"
    PUBMED = "PubMed"


class Split(Enum):
    PUBLIC = "public"
    COMPLETE = "complete"


def get_adj_matrix(edge_index: torch.Tensor, num_nodes: int) -> torch.FloatTensor:
    """
    Construct dense ADJ matrix from edge indexes.
    """
    # Convert edge_indexes to scipy sparse matrix.
    adj_matrix = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0].numpy(), edge_index[1].numpy())),
                               shape=(num_nodes, num_nodes), dtype=np.float3)
    return torch.FloatTensor(adj_matrix.todense())


def get_planetoid_dataset(dataset_name: PlanetoidDatasetName, normalize_features=False,
                          split=Split.PUBLIC, construct_dense_adj_matrix: bool = True) -> Planetoid:
    """
    Get the planetoid dataset which is capable for loading ['Cora', 'CiteSeer', 'PubMed'] datasets. The original code is
    from "https://github.com/russellizadi/ssp/blob/master/experiments/datasets.py".
    :param dataset_name: Dataset name
    :param normalize_features:
    :param split: Whether to normalize the dataset.
    :param construct_dense_adj_matrix: Whether to create the dense ADJ matrix.
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

    if normalize_features:
        dataset.transform = T.NormalizeFeatures()

    if construct_dense_adj_matrix:
        dataset[0].adj_matrix = get_adj_matrix(dataset[0].edge_index, dataset[0].num_nodes)

    return dataset
