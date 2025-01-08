import inspect

import torch
from torch import nn
from torch_geometric.nn import GCNConv

from gnn.models.base_network import BaseNetwork
from gnn.models.networks.deep_rp_robust_gcn import RanPACLayer
from gnn.models.networks.robust_gcn import NodeSelfAtten


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
        self.conv1 = GCNConv(input_dim, net_size)
        self.conv2 = GCNConv(net_size, net_size)
        self.conv_out = GCNConv(net_size, output_dim)

        self.self_atten = NodeSelfAtten(self.net_size)
        self.rp_final = RanPACLayer(self.net_size, self.net_size, self.lambda_value)
        self.classifier = nn.Linear(self.net_size, output_dim)

    def forward(self, inputs: torch.Tensor, efficient_mode=True):
        g1 = self.conv1(inputs)
        g2 = self.conv2(g1)
        g_out = self.conv_out(g2)
        att_out = self.self_atten(g_out)
        rp_out = self.rp_final(att_out)
        preds = self.classifier(rp_out)
        return preds
