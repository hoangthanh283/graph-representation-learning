from typing import Any, Dict, List, NamedTuple, Optional, Union

import munch
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops

from gnn.data_generator import augmentor, data_process
from gnn.data_generator.augmentor import BaseAugmentor
from gnn.data_generator.datasets.base_dataset import BaseDataset
from gnn.utils.logger.color_logger import color_logger


class GraphDataDetails(NamedTuple):
    """
    Named tuple for the graph data details.
    """
    nodes: torch.Tensor
    labels: torch.Tensor
    adj_matrix: torch.Tensor
    classes: List[int]


class PlanetoidDataset(BaseDataset):
    def __init__(self, data_config: munch.munchify, **kwargs: Dict[str, Any]):
        """Initialize the Planetoid datasets from torch geometric

        Args:
            config: Configuration parameters.
            **kwargs: Additional keyword arguments.
        """
        super(PlanetoidDataset, self).__init__()
        self.data_config = data_config
        self.logger = color_logger(__name__, testing_mode=False)

        # Load dataset samples.
        self.graph_dataset = self._create_dataset(self.data_config.dataset_type, self.data_config.dataset_name,
                                                  self.data_config.data_dir)
        self.num_samples = len(self.graph_dataset)
        self.graph_data = self._load_graph_data(self.graph_dataset)
        # Get the class names and assign an unique integer to each of them
        self.classes = self.graph_data.classes
        self.num_class = len(self.classes)
        # Loading all data processors.
        self.data_processors = self._load_data_processors()
        self.logger.info(f"Initialize {self.data_config.dataset_name} dataset, loading {self.num_samples} samples...")

    def _create_dataset(self, dataset_type: str, dataset_name: Optional[str] = None, cache_path: str = "/tmp"
                        ) -> torch_geometric.datasets:
        """Load dataset from the torch_geometric datasets."""
        dataset_class = getattr(torch_geometric.datasets, dataset_type)
        dataset = dataset_class(root=cache_path, name=dataset_name, split="full") if dataset_name else dataset_class(root=cache_path)
        dataset.transform = T.NormalizeFeatures()
        return dataset

    def _load_graph_data(self, dataset: torch_geometric.datasets) -> GraphDataDetails:
        # Normalize features (row-wise normalization as done in GraphSAGE).
        nodes = dataset.x
        labels = dataset.y
        adj_matrix = self._create_adjacency_matrix(dataset.edge_index, nodes.size(0))
        classes = sorted(dataset.y.unique().tolist())  # Indices of the classes (Sorted ascending order by default).
        return GraphDataDetails(nodes, labels, adj_matrix, classes)

    def _create_adjacency_matrix(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Create adjacency matrix from edge index."""
        # Add self-loops
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)[0]

        # Ensure edge indices are within the valid range
        edge_index = edge_index[:, (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)]

        # Create sparse adjacency matrix
        values = torch.ones(edge_index.size(1))
        adj_matrix = torch.sparse_coo_tensor(
            edge_index,
            values,
            size=(num_nodes, num_nodes)
        )
        return adj_matrix

    def _load_data_processors(self) -> List[Union[BaseAugmentor, data_process.BaseDataProcess]]:
        """Load all data processors (augmentations/preprocessing/..etc)."""
        data_processors: List[BaseAugmentor] = []
        if self.data_config.augmentations:
            # Add augmentation methods.
            for aug in self.data_config.augmentations:
                aug_process = getattr(augmentor, aug)._from_config(self.data_config.augmentations[aug])
                self.logger.info(f"Type data processor: {aug_process.__class__.__name__}")
                data_processors.append(aug_process)

        if self.data_config.data_process:
            # Add preprocessing methods.
            for pre in self.data_config.data_process:
                pre_process = getattr(data_process, pre)._from_config(self.data_config.data_process[pre])
                self.logger.info(f"Type data processor: {pre_process.__class__.__name__}")
                data_processors.append(pre_process)
        return data_processors

    def __getitem__(self, index: int, retry: int = 0) -> Dict[str, Any]:
        sample = {
            "labels": self.graph_data.labels,
            "nodes": self.graph_data.nodes,
            # "adj_matrix": self.graph_data.adj_matrix
            "edge_index": self.graph_dataset.edge_index,
        }
        for data_processor in self.data_processors:
            sample = data_processor(sample)
        return sample

    def __len__(self) -> int:
        return self.num_samples


if __name__ == "__main__":
    reddit_dataset = PlanetoidDataset(
        munch.munchify(
            {"dataset_name": "Reddit", "data_dir": "./assets", "augmentations": [], "data_process": []}
        )
    )
