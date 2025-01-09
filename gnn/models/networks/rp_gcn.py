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
from gnn.models.networks.robust_gcn import NodeSelfAtten


class RanPACLayer(nn.Module):
    def __init__(self, input_dim, output_dim, lambda_value: Optional[float] = None):
        super(RanPACLayer, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        # Freeze the random projection matrix
        for param in self.projection.parameters():
            param.requires_grad = False

        # Initialize weights with a normal distribution
        init.normal_(self.projection.weight, mean=0, std=1.0)
        # self.projection.weight.data.uniform_(-1, 1)
        self.projection.weight *= np.sqrt(output_dim)
        if lambda_value:
            self.projection.weight *= lambda_value

    def forward(self, x, lambda_param: Optional[Union[float, torch.FloatTensor]] = None,
                lambda_bias: Optional[Union[float, torch.FloatTensor]] = None) -> torch.Tensor:
        x = self.projection(x)
        if lambda_param:
            x *= lambda_param
        if lambda_bias:
            x += lambda_bias
        x = F.leaky_relu(x, negative_slope=0.2)
        return x


class GCNBlock(nn.Module):
    """
    A single block of GCN which combines GraphConv, BatchNorm, and LeakyReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(GCNBlock, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, A):
        x = self.gcn(x, A)
        x = self.bn(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        return x


class EmbeddingBlock(nn.Module):
    """
    A single block of Embedding which combines Linear, BatchNorm, and LeakyReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(EmbeddingBlock, self).__init__()
        self.emb = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.emb(x)
        x = x.permute(0, 2, 1)  # Permute the input data to (N, C, L)
        x = self.bn(x)
        x = x.permute(0, 2, 1)  # (Optional) Permute back to original shape.
        x = F.leaky_relu(x, negative_slope=0.2)
        return x


class RPGCN(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, net_size: int = 16, dropout_rate: float = 0.5,
                 lambda_value: Optional[float] = None):
        super(RPGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, net_size) 
        self.p = dropout_rate
        if lambda_value:
            # self.lambda_param = Parameter(torch.FloatTensor([-0.43]))
            # self.lambda_bias = Parameter(torch.FloatTensor([0.0]))

            self.lambda_param = Parameter(torch.FloatTensor(1))
            self.lambda_bias = Parameter(torch.FloatTensor(1))
        else:
            self.lambda_param = None
            self.lambda_bias = None
        self.rp_final = RanPACLayer(net_size, net_size)
        self.conv2 = GCNConv(net_size, output_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        init.xavier_uniform_(self.conv1.lin.weight)
        init.xavier_uniform_(self.conv2.lin.weight)
        if self.lambda_param and self.lambda_bias:
            stdv = 1. / math.sqrt(self.lambda_param.size(0))
            self.lambda_param.data.uniform_(-stdv, stdv)
            self.lambda_bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        x, edge_index = inputs
        if self.lambda_param and self.lambda_bias:
            x = self.rp_final(self.conv1(x, edge_index), self.lambda_param, self.lambda_bias)
        else:
            x = F.leaky_relu(self.conv1(x, edge_index), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x, edge_index), negative_slope=0.2)
        return x
