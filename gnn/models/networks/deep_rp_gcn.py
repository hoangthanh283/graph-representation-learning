import inspect

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gnn.models.base_network import BaseNetwork
from gnn.models.networks.robust_gcn import GraphConv

NUM_GCN_LAYERS = 29
# How the RP layer should be placed in the network after how many GCN layers.
RP_LAYER_RELATIVE_POSITION = None # 3
SKIP_CONNECTION_POS = 3


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


class CustomGCNSequential(nn.Module):
    def __init__(self, net_size, num_edges, lambda_value):
        super(CustomGCNSequential, self).__init__()
        # self.dropout = nn.Dropout(p=0.3)
        # self.edge_dropout = nn.Dropout(p=0.2)
        self.layers = nn.ModuleList()
        for ii in range(NUM_GCN_LAYERS):
            if SKIP_CONNECTION_POS and ii % SKIP_CONNECTION_POS == 0:
                self.layers.append(GCNBlock(net_size * 2, net_size, num_edges))
            else:
                self.layers.append(GCNBlock(net_size, net_size, num_edges))
            if RP_LAYER_RELATIVE_POSITION and ii % RP_LAYER_RELATIVE_POSITION == 0:
                self.layers.append(RanPACLayer(net_size, net_size, lambda_value))

    def forward(self, feats, A, preprocess_A):
        # Pass the inputs through all layers sequentially
        prev_feats = feats
        for idx, layer in enumerate(self.layers):
            if (idx + SKIP_CONNECTION_POS) % SKIP_CONNECTION_POS == 0:
                prev_feats = feats
 
            if isinstance(layer, RanPACLayer):
                feats = layer(feats)
            else:
                if SKIP_CONNECTION_POS and idx % SKIP_CONNECTION_POS == 0:
                    feats = torch.cat([prev_feats, feats], dim=-1)
                    # A = self.edge_dropout(A)

            feats = layer(feats, A, preprocess_A)
            # if SKIP_CONNECTION_POS and idx % SKIP_CONNECTION_POS == 0:
            #     feats = self.dropout(feats)
        return feats


class DeepRPGCN(BaseNetwork):
    def __init__(self, input_dim, output_dim, num_edges, net_size=256, rp_size=10000,
                 lambda_value=0.01):
        super(DeepRPGCN, self).__init__()
        # Use inspect to get the current frame's arguments.
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        init_params = {arg: values[arg] for arg in args if arg != "self"}
        self.logger.info(f"Initialized with parameters: {init_params}")

        self.output_dim = output_dim
        self.net_size = net_size
        self.rp_size = rp_size
        self.lambda_value = lambda_value
        self.dropout = nn.Dropout(p=0.3)

        self.emb1 = EmbeddingBlock(input_dim, self.net_size)
        self.gcn_layers = CustomGCNSequential(self.net_size, num_edges, lambda_value)
        self.emb2 = EmbeddingBlock(self.net_size, self.net_size)
        # self.rp_embed2 = RanPACLayer(self.net_size, self.net_size, lambda_value)
        self.classifier = nn.Linear(self.net_size, output_dim)

    def forward(self, inputs, efficient_mode=True):
        V, A = inputs
        A = A.permute(0, 1, 3, 2)
        embedding = self.emb1(V)

        # Preprocess A before feeding to GCN. Therefore, we only preprocess adjacency matrix once.
        if efficient_mode:
            A = self.gcn_layers.layers[0].gcn.preprocess_adj(A)
            preprocess_A = False
        else:
            # Preprocess A at each layer.
            preprocess_A = True

        # N GraphConv layers.
        g_n = self.gcn_layers(embedding, A, preprocess_A)

        # 2nd Embedding layer
        feats = self.emb2(g_n)
        # feats = self.rp_embed2(feats, self.lambda_value)

        # Final classifier
        feats = self.dropout(feats)
        outputs = self.classifier(feats)
        return outputs
