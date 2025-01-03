import inspect

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gnn.models.base_network import BaseNetwork
from gnn.models.networks.robust_gcn import GraphConv, NodeSelfAtten


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
    def __init__(self, in_channels, out_channels, num_edges):
        super(GCNBlock, self).__init__()
        self.gcn = GraphConv(in_channels, out_channels, num_edges)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, A, preprocess_A):
        x = self.gcn(x, A, preprocess_A)
        x = x.permute(0, 2, 1)  # Permute the input data to (N, C, L)
        x = self.bn(x)
        x = x.permute(0, 2, 1)  # (Optional) Permute back to original shape.
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


class DeepRPRobustGCN(BaseNetwork):
    """ Deep Robust GCN with Random Projection Layer. """
    def __init__(self, input_dim, output_dim, num_edges, net_size=256, use_attention=True, rp_size=10000,
                 lambda_value=0.01):
        super(DeepRPRobustGCN, self).__init__()
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

        self.dropout = nn.Dropout(p=0.3)
        self.edge_dropout = nn.Dropout(p=0.2)

        self.emb1 = EmbeddingBlock(input_dim, self.net_size)
        self.gcn1 = GCNBlock(self.net_size, self.net_size, num_edges)
        # self.rp_1 = RanPACLayer(self.net_size, self.net_size, self.lambda_value)

        self.gcn2 = GCNBlock(self.net_size, self.net_size, num_edges)
        self.gcn3 = GCNBlock(self.net_size * 2, self.net_size, num_edges)
        self.gcn4 = GCNBlock(self.net_size, self.net_size, num_edges)
        # self.rp_4 = RanPACLayer(self.net_size, self.net_size, self.lambda_value)

        self.gcn5 = GCNBlock(self.net_size, self.net_size, num_edges)
        self.gcn6 = GCNBlock(self.net_size * 2, self.net_size, num_edges)
        self.gcn7 = GCNBlock(self.net_size, self.net_size, num_edges)
        # self.rp_7 = RanPACLayer(self.net_size, self.net_size, self.lambda_value)

        self.gcn8 = GCNBlock(self.net_size, self.net_size, num_edges)
        self.gcn9 = GCNBlock(self.net_size, self.net_size, num_edges)
        self.emb2 = EmbeddingBlock(self.net_size * 2, self.net_size)
        # self.rp_embed2 = RanPACLayer(self.net_size, self.net_size, self.lambda_value)
        self.rp_embed2 = RanPACLayer(self.net_size, self.net_size, 1)
        self.self_atten = NodeSelfAtten(self.net_size)

        # self.rp_final = RanPACLayer(self.net_size, self.net_size, self.lambda_value)
        self.classifier = nn.Linear(self.net_size, output_dim)

    def forward(self, inputs, efficient_mode=True):
        V, A = inputs
        A = A.permute(0, 1, 3, 2)
        # embedding = self.dropout(self.emb1(V))
        embedding = self.emb1(V)

        # Preprocess A before feeding to GCN. Therefore, we only preprocess adjacency matrix once.
        if efficient_mode:
            A = self.gcn1.gcn.preprocess_adj(A)
            preprocess_A = False
        else:
            # Preprocess A at each layer.
            preprocess_A = True

        # 1st GraphConv
        g1 = self.gcn1(embedding, A, preprocess_A)
        # g1 = self.rp_1(g1)

        # Second GraphConv
        g2 = self.gcn2(g1, A, preprocess_A)

        # 2nd GraphConv
        g3 = self.gcn3(torch.cat([g1, g2], dim=-1), self.edge_dropout(A), preprocess_A)
        g3 = self.dropout(g3)

        # 4th GraphConv
        g4 = self.gcn4(g3, A, preprocess_A)
        # g4 = self.rp_4(g4)

        # 5th GraphConv
        g5 = self.gcn5(g4, A, preprocess_A)

        # 6th GraphConv
        g6 = self.gcn6(torch.cat([g4, g5], dim=-1), self.edge_dropout(A), preprocess_A)
        g6 = self.dropout(g6)

        # 7th GraphConv
        g7 = self.gcn7(g6, A, preprocess_A)
        # g7 = self.rp_7(g7)

        # 8th GraphConv
        g8 = self.gcn8(g7, self.edge_dropout(A), preprocess_A)

        # 9th GraphConv
        g9 = self.gcn9(g8, self.edge_dropout(A), preprocess_A)
        g9 = self.dropout(g9)

        # 2nd Embedding layer
        feats = self.emb2(torch.cat([g8, g9], dim=-1))
        feats = self.rp_embed2(feats, self.lambda_value)

        # Attention layer
        feats = self.self_atten(feats)
        # feats = self.rp_final(feats)

        # Final classifier
        feats = self.dropout(feats)
        outputs = self.classifier(feats)
        return outputs
