import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gnn.models.base_network import BaseNetwork
from gnn.models.networks.robust_gcn import (GraphConv, NodeSelfAtten,
                                            make_linear_relu)

RP_FACTOR = 10


class RanPACLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RanPACLayer, self).__init__()
        # Initialize a random projection matrix
        self.projection = nn.Linear(input_dim, output_dim, bias=False)

        # Freeze the random projection matrix
        for param in self.projection.parameters():
            param.requires_grad = False

        # Initialize weights with a normal distribution
        # nn.init.normal_(self.projection.weight, mean=0, std=1.0 / output_dim**0.5)
        nn.init.normal_(self.projection.weight, mean=0, std=1.0)

    def forward(self, x):
        return self.projection(x)


class GraphCNNDropEdge(BaseNetwork):
    def __init__(self, input_dim, output_dim, num_edges, net_size=256, use_attention=True):
        super(GraphCNNDropEdge, self).__init__()
        self.output_dim = output_dim
        self.net_size = net_size
        self.emb1 = make_linear_relu(input_dim, self.net_size)
        self.dropout = nn.Dropout(p=0.5)
        self.edge_dropout = nn.Dropout(p=0.3)
        self.gcn1 = GraphConv(self.net_size, self.net_size, num_edges)
        self.gcn2 = GraphConv(self.net_size, self.net_size, num_edges)
        self.gcn3 = GraphConv(self.net_size * 2, self.net_size, num_edges)

        half_net_size = self.net_size // 2
        self.emb2 = make_linear_relu(self.net_size * 2, half_net_size)

        # rp_size = half_net_size * RP_FACTOR
        # self.w_rand = torch.randn(half_net_size, rp_size).cuda() * np.sqrt(rp_size)
        # self.w_rand = RanPACLayer(half_net_size, half_net_size)
        self.use_attention = use_attention
        if use_attention:
            self.self_atten = NodeSelfAtten(half_net_size)

        # self.gcn4 = GraphConv(half_net_size, half_net_size, num_edges)
        # self.gcn5 = GraphConv(half_net_size, half_net_size, num_edges)

        rp_size = half_net_size * RP_FACTOR
        # self.w_rand = torch.randn(half_net_size, rp_size).cuda()  # * np.sqrt(rp_size)
        self.w_rand = RanPACLayer(half_net_size, rp_size)
        self.classifier = nn.Linear(rp_size, output_dim)

    def forward(self, inputs, efficient_mode=True):
        V, A = inputs
        A = A.permute(0, 1, 3, 2)
        embedding = self.dropout(self.emb1(V))

        # Preprocess A before feeding to GCN
        # Therefore, we only preprocess adjacency matrix once
        if efficient_mode:
            A = self.gcn1.preprocess_adj(A)
            preprocess_A = False
        else:
            # Preprocess A at each layer
            preprocess_A = True

        # First GraphConv
        g1 = F.relu(self.gcn1(embedding, self.edge_dropout(A), preprocess_A))
        g1 = self.dropout(g1)

        # Second GraphConv
        g2 = F.relu(self.gcn2(g1, self.edge_dropout(A), preprocess_A))
        g2 = self.dropout(g2)

        # Third GraphConv
        new_v = torch.cat([g1, g2], dim=-1)
        g3 = F.relu(self.gcn3(new_v, self.edge_dropout(A), preprocess_A))
        g3 = self.dropout(g3)

        new_v = torch.cat([g1, g3], dim=-1)
        new_v = self.emb2(new_v)

        # new_v = F.relu(self.w_rand(new_v))
        # new_v = F.relu(new_v @ self.w_rand)

        if self.use_attention:
            new_v = self.self_atten(new_v)

        # Apply the random projection layter.
        # new_v = F.relu(new_v @ self.w_rand)
        new_v = F.relu(self.w_rand(new_v))
        new_v = self.dropout(new_v)

        # Final feature extractor.
        return self.classifier(new_v)
