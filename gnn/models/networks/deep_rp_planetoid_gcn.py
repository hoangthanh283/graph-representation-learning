import inspect

import torch
from torch import nn

from gnn.models.base_network import BaseNetwork
from gnn.models.networks.deep_rp_robust_gcn import RanPACLayer


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    Apply normalization for adj matrix.
    :param adj: Adjacency matrix.
    :return: Normalized adjacency matrix.
    """
    deg = torch.sum(adj, dim=1, keepdim=True)
    deg = torch.pow(deg, -0.5)
    adj_norm = adj * deg * deg.transpose(-1, -2)
    return adj_norm


class PlanetoidGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PlanetoidGraphConv, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # Todo: init the weight
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x, adj):
        """
        :param x: (num_nodes, input_dim)
        :param adj: (num_nodes, num_nodes)
        """
        support = self.linear(x)
        output = torch.matmul(adj, support)
        return output

    def __repr__(self):
        return "PlanetoidGraphConv"


class DeepRPPlanetoidGCN(BaseNetwork):
    """ Deep Robust GCN with Random Projection Layer. """

    def __init__(self, input_dim, output_dim, net_size=256, use_attention=True, rp_size=10000,
                 lambda_value=0.01):
        super(DeepRPPlanetoidGCN, self).__init__()
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

        # Conv blocks.
        self.conv1 = PlanetoidGraphConv(input_dim, net_size)
        self.conv2 = PlanetoidGraphConv(net_size, net_size)
        self.conv_out = PlanetoidGraphConv(net_size, net_size)

        self.rp_final = RanPACLayer(self.net_size, self.net_size, self.lambda_value)
        self.classifier = nn.Linear(self.net_size, output_dim)

    def forward(self, inputs: torch.Tensor, efficient_mode=True):
        x, adj = inputs
        adj_norm = normalize_adj(adj)
        g_out = self.conv1(x, adj_norm)
        g_out = self.conv2(g_out, adj_norm)
        g_out = self.conv_out(g_out, adj_norm)
        # g_out = self.rp_final(g_out)
        preds = self.classifier(g_out)
        return preds
