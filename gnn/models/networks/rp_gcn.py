import math
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge
from gnn.models.base_network import BaseNetwork


class RanPACLayer(nn.Module):
    def __init__(self, input_dim, output_dim, lambda_value: Optional[float] = None):
        super(RanPACLayer, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        # Freeze the random projection matrix
        for param in self.projection.parameters():
            param.requires_grad = False

        # Initialize weights with a normal distribution
        init.normal_(self.projection.weight, mean=0, std=1.0)
        if lambda_value:
            self.projection.weight *= (np.sqrt(output_dim) * lambda_value)

    def forward(self, x, lambda_param: Optional[Union[float, torch.FloatTensor]] = None,
                lambda_bias: Optional[Union[float, torch.FloatTensor]] = None) -> torch.Tensor:
        x = self.projection(x)
        if lambda_param:
            x *= lambda_param
        if lambda_bias:
            x += lambda_bias
        # x = F.leaky_relu(x, negative_slope=0.2)
        x = F.relu(x)
        return x


class RPGCN(BaseNetwork):
    def __init__(self, input_dim: int, output_dim: int, net_size: int = 16, dropout_rate: float = 0.5,
                 use_rp: bool = False, lambda_value: Optional[float] = None):
        super(RPGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, net_size)
        self.p = dropout_rate
        self.use_rp = use_rp
        if self.use_rp:
            self.rp_final = RanPACLayer(net_size, net_size)
            # self.lambda_value = lambda_value if lambda_value else Parameter(torch.FloatTensor(1)) # -0.43, -0.57.
            # self.lambda_value = Parameter(torch.FloatTensor(1))
            # self.lambda_bias = Parameter(torch.FloatTensor(1))
            self.lambda_value = 0.2
            self.lambda_bias = 0.0
        else:
            self.lambda_value = None
            self.lambda_bias = None
        self.conv2 = GCNConv(net_size, output_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        init.xavier_uniform_(self.conv1.lin.weight)
        init.xavier_uniform_(self.conv2.lin.weight)

    def forward(self, inputs):
        x, edge_index = inputs
        x = F.relu(self.conv1(x, edge_index))
        if self.use_rp:
            x = self.rp_final(x, self.lambda_value, self.lambda_bias)
            # x = self.rp_final(x)
        # x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x
