import inspect

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gnn.models.base_network import BaseNetwork
from gnn.models.networks.robust_gcn import (GraphConv, NodeSelfAtten,
                                            make_linear_relu)


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

    def forward(self, x):
        return self.projection(x)


class RPRobustFilterGraphCNNDropEdge(BaseNetwork):
    def __init__(self, input_dim, output_dim, num_edges, net_size=256, use_attention=True, rp_size=10000,
                 lambda_value=0.05):
        super(RPRobustFilterGraphCNNDropEdge, self).__init__()
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
        self.emb1 = make_linear_relu(input_dim, self.net_size)
        self.dropout = nn.Dropout(p=0.5)
        self.edge_dropout = nn.Dropout(p=0.3)

        self.gcn1 = GraphConv(self.net_size, self.net_size, num_edges)
        # self.rp_1 = RanPACLayer(self.net_size, self.rp_size, self.lambda_value)
        # self.rp_1 = torch.randn(self.net_size, self.net_size).cuda() * np.sqrt(self.net_size) * self.lambda_value

        self.gcn2 = GraphConv(self.net_size, self.net_size, num_edges)
        # self.rp_2 = torch.randn(self.net_size, self.net_size).cuda() * np.sqrt(self.net_size) * self.lambda_value

        self.gcn3 = GraphConv(self.net_size * 2, self.net_size, num_edges)
        # self.rp_3 = torch.randn(self.net_size, self.net_size).cuda() * np.sqrt(self.net_size) * self.lambda_value
        # self.rp_3 = RanPACLayer(self.net_size, self.net_size, self.lambda_value)

        self.half_net_size = self.net_size // 2
        self.emb2 = make_linear_relu(self.net_size * 2,  self.half_net_size)
        self.rp_emb = RanPACLayer(self.half_net_size, self.rp_size, self.lambda_value)
        if use_attention:
            self.self_atten = NodeSelfAtten(self.rp_size)

        # self.rp_final = torch.randn(self.half_net_size, self.rp_size).cuda()
        # self.rp_final = torch.randn(self.half_net_size, self.half_net_size).cuda() * np.sqrt(self.net_size) * self.lambda_value
        self.rp_final = RanPACLayer(self.rp_size, self.rp_size, self.lambda_value)
        self.classifier = nn.Linear(self.rp_size, output_dim)
        # self.classifier = nn.Linear(self.half_net_size, output_dim)

    def forward(self, inputs, efficient_mode=True):
        V, A = inputs
        A = A.permute(0, 1, 3, 2)
        embedding = self.dropout(self.emb1(V))

        # Preprocess A before feeding to GCN. Therefore, we only preprocess adjacency matrix once.
        if efficient_mode:
            A = self.gcn1.preprocess_adj(A)
            preprocess_A = False
        else:
            # Preprocess A at each layer.
            preprocess_A = True

        # First GraphConv
        g1 = F.relu(self.gcn1(embedding, self.edge_dropout(A), preprocess_A))
        g1 = self.dropout(g1)
        # g1 = F.leaky_relu(g1 @ self.rp_1)
        # g1 = F.leaky_relu(self.rp_1(g1))

        # Second GraphConv
        g2 = F.relu(self.gcn2(g1, self.edge_dropout(A), preprocess_A))
        g2 = self.dropout(g2)
        # g2 = F.leaky_relu(g2 @ self.rp_2)

        new_v = torch.cat([g1, g2], dim=-1)

        # Third GraphConv
        g3 = F.relu(self.gcn3(new_v, self.edge_dropout(A), preprocess_A))
        g3 = self.dropout(g3)
        # g3 = F.leaky_relu(g3 @ self.rp_3)
        # g3 = F.leaky_relu(self.rp_3(g3))

        new_v = torch.cat([g1, g3], dim=-1)
        new_v = self.emb2(new_v)
        # new_v = F.leaky_relu(new_v @ self.rp_emb)
        new_v = F.leaky_relu(self.rp_emb(new_v))

        if self.use_attention:
            new_v = self.self_atten(new_v)

        # new_v = F.relu(torch.matmul(new_v, self.rp_final))
        # new_v = F.relu(new_v @ self.rp_final)
        new_v = F.leaky_relu(self.rp_final(new_v))
        new_v = self.dropout(new_v)
        return self.classifier(new_v)
