from enum import Enum
from pathlib import Path

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


def get_planetoid_dataset(dataset_name: PlanetoidDatasetName, normalize_features=False,
                          split=Split.PUBLIC) -> Planetoid:
    """
    Get the planetoid dataset which is capable for loading ['Cora', 'CiteSeer', 'PubMed'] datasets. The original code is
    from "https://github.com/russellizadi/ssp/blob/master/experiments/datasets.py".
    :param dataset_name: Dataset name
    :param normalize_features:
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

    if normalize_features:
        dataset.transform = T.NormalizeFeatures()

    return dataset
