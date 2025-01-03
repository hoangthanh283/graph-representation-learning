import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from gnn.models.base_network import BaseNetwork


def knn(x: Tensor, K: int):
    """
    Indices of K nearest neighbors
    TODO: Combine masking
    Parameters
    ---------
    x: Tensor shape (B, V, F)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=K, dim=-1)[1]  # (B, V, K)
    return idx


def get_graph_feature(x: Tensor, K: int = 20, idx: Tensor = None):
    """
    Get the graph feature using K nearest neightbors
    Implemented using equation (7) at https://arxiv.org/pdf/1801.07829v2.pdf
    Parameters
    ----------
    x : tensor shape (B, F, V)
    K : KNN
    idx: idexing tensor to use, if not provided, KNN indeices are used
    Returns
    -------
    feature: tensor shape (B, F, V, K)
    """
    B = x.size(0)
    V = x.size(2)
    x = x.view(B, -1, V)

    # Use the maximum number of vertices when K > V
    K = min(K, V)

    # Get K Nearest Neighbors
    if idx is None:
        idx = knn(x, K=K)  # (B, V, K)

    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * V

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(B * V, -1)[idx, :]
    feature = feature.view(B, V, K, num_dims)
    x = x.view(B, V, 1, num_dims).repeat(1, 1, K, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class DGCNN(BaseNetwork):
    def __init__(self, in_channels: int, out_channels: int, kk: int):
        super(DGCNN, self).__init__()
        self.kk = kk
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * self.in_channels, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, self.out_channels, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        x: tensor shape (B, V, F)
        """
        xx, _ = inputs
        xx = xx.permute(0, 2, 1).contiguous()  # (B, F, V)
        xx = get_graph_feature(xx, K=self.kk)
        xx = self.conv1(xx)
        x1 = xx.max(dim=-1, keepdim=False)[0]

        xx = get_graph_feature(x1, K=self.kk)
        xx = self.conv2(xx)
        x2 = xx.max(dim=-1, keepdim=False)[0]

        xx = get_graph_feature(x2, K=self.kk)
        xx = self.conv3(xx)
        x3 = xx.max(dim=-1, keepdim=False)[0]

        xx = get_graph_feature(x3, K=self.kk)
        xx = self.conv4(xx)
        x4 = xx.max(dim=-1, keepdim=False)[0]

        xx = torch.cat((x1, x2, x3, x4), dim=1)
        xx = self.conv5(xx)
        xx = xx.permute(0, 2, 1).contiguous()  # (B, F, V)
        return xx
