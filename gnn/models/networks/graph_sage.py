from typing import Tuple

import torch
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv
from gnn.models.base_network import BaseNetwork


class GraphSage(BaseNetwork):
    def __init__(self, input_dim: int, output_dim: int, net_size: int, dropout_rate: float = 0.5):
        super(GraphSage, self).__init__()
        """Graph Sage Network from https://arxiv.org/abs/1706.02216 for Node classification task.

        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output features.
            net_size (int): Dimension of hidden features.
            dropout_rate (float, optional): Dropout rate (default: 0.5).
        """
        self.dropout_rate = dropout_rate
        self.conv1 = SAGEConv(input_dim, net_size)
        self.conv2 = SAGEConv(net_size, output_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, edge_index = inputs
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x
