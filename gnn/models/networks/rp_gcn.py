import inspect

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge

from gnn.models.base_network import BaseNetwork
from gnn.models.networks.robust_gcn import NodeSelfAtten


class RanPACLayer(nn.Module):
    def __init__(self, input_dim, output_dim, lambda_value):
        super(RanPACLayer, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        # Freeze the random projection matrix
        for param in self.projection.parameters():
            param.requires_grad = False

        # Initialize weights with a normal distribution
        nn.init.normal_(self.projection.weight, mean=0, std=1.0)
        self.projection.weight *= (np.sqrt(output_dim) * lambda_value)

    def forward(self, x, lambda_value=1):
        x = self.projection(x) * lambda_value
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


class RPGCN(BaseNetwork):
    """Graph Convolution Network with Random Projection Layer. """
    def __init__(self, input_dim, output_dim, net_size=256, use_attention=True, rp_size=10000,
                 lambda_value=0.01):
        super(RPGCN, self).__init__()
        # Use inspect to get the current frame's arguments.
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        init_params = {arg: values[arg] for arg in args if arg != "self"}
        self.logger.info(f"Initialized with parameters: {init_params}")

        self.output_dim = output_dim
        self.net_size = net_size
        self.rp_size = rp_size
        self.lambda_value = lambda_value
        self.use_attention = use_attention

        self.dropout = nn.Dropout(p=0.5)
        self.gcn1 = GCNBlock(input_dim, self.net_size)
        self.gcn2 = GCNBlock(self.net_size, self.net_size)
        self.rp_final = RanPACLayer(self.net_size, self.net_size, 1)
        if self.use_attention:
            self.self_atten = NodeSelfAtten(self.net_size)
        self.classifier = nn.Linear(self.net_size, output_dim)
        # self.classifier = GCNBlock(self.net_size, output_dim)

    def forward(self, inputs):
        vertices, edge_index = inputs
        # 1st GraphConv
        feats = self.gcn1(vertices, edge_index)
        feats = self.dropout(feats)
        # edge_index, _ = dropout_edge(edge_index, p=0.2)

        # 2nd GraphConv
        feats = self.gcn2(feats, edge_index)
        # feats = self.rp_final(feats, self.lambda_value)
        if self.use_attention:
            # Attention layer
            feats = self.self_atten(feats)

        # Final classifier
        feats = self.dropout(feats)
        outputs = self.classifier(feats)
        return outputs
