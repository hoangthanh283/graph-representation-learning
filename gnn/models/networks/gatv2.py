import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import leaky_relu

from gnn.models.base_network import BaseNetwork

logger = logging.getLogger(__name__)


class Norm(nn.Module):
    def __init__(self, input_dimention, bn=False):
        super().__init__()
        self.bn = bn
        if bn:
            self.norm = nn.BatchNorm1d(input_dimention)
        else:
            self.norm = nn.LayerNorm(input_dimention)
        self.relu = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        if self.bn:
            output = self.norm(x.permute(0, 2, 1))
            output = self.relu(output.permute(0, 2, 1))
        else:
            output = self.norm(x)
            output = self.relu(output)
        return output


class MakeParameterW(nn.Module):
    def __init__(self, in_features, out_features):
        super(MakeParameterW, self).__init__()
        self.parameter = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        torch.nn.init.xavier_uniform_(self.parameter)

    def forward(self):
        return self.parameter


class MakeParameterA(nn.Module):
    def __init__(self, out_features):
        super(MakeParameterA, self).__init__()
        self.parameter = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        torch.nn.init.xavier_uniform_(self.parameter)

    def forward(self):
        return self.parameter


class GraphAttentionLayer(nn.Module):
    def __init__(self, no_A, in_features, out_features, dropout, multi_head=4, ratio=8):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_squeeze = out_features // ratio
        self.no_A = no_A
        self.W = nn.ModuleList()
        self.A = nn.ModuleList()
        self.norm = nn.ModuleList()
        for _ in range(no_A + 1):
            self.W.append(MakeParameterW(self.in_features, self.out_features_squeeze))
            self.A.append(MakeParameterA((self.out_features_squeeze)))
            self.norm.append(Norm(self.out_features_squeeze))

        if in_features != out_features:
            self.map = nn.Linear(in_features, out_features, True)
        else:
            self.map = nn.Identity()

        self.squeeze = nn.Linear(
            self.out_features_squeeze * (no_A + 1), out_features, True
        )

    def forward(self, V, adj):

        n_node = V.size()[1]
        batch_size = V.size()[0]
        output = torch.zeros(size=(batch_size, n_node, self.out_features))

        for num_A in range(self.no_A + 1):

            h = torch.matmul(V, self.W[num_A]())

            e = torch.cat(
                [
                    h.repeat(1, 1, n_node).view(batch_size, n_node * n_node, -1),
                    h.repeat(1, n_node, 1),
                ],
                dim=1,
            ).view(batch_size, n_node, n_node, -1)

            # (batch_size, n_node, n_node, out_features*2) -> (batch_size, n_node, n_node,1)  -> (batch_size, n_node, n_node)
            e = torch.matmul(e, self.A[num_A]()).squeeze(-1)
            e = leaky_relu(e, inplace=True)

            if num_A < self.no_A:
                num_adj = adj[:, :, num_A, :]
            else:
                # == torch.eye(n_node).repeat(batch_size, 1).reshape(batch_size, n_node, n_node).to(adj.device)
                num_adj = torch.eye(n_node).to(adj.device)

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(num_adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout, training=self.training)
            attention = torch.matmul(attention, h)
            attention = self.norm[num_A](attention)

            if num_A == 0:
                output = attention
            else:
                output = torch.cat((output, attention), -1)

        output = self.squeeze(output)
        output += self.map(V)

        return output, adj

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class MakeParameterAV2(nn.Module):
    def __init__(self, multi_head, out_features):
        super(MakeParameterAV2, self).__init__()
        self.parameter = nn.Parameter(
            torch.zeros(size=(1, 1, 1, multi_head, out_features))
        )
        torch.nn.init.xavier_uniform_(self.parameter)

    def forward(self):
        return self.parameter


# GATs2 - How Attentive are Graph Attention Networks?
# paper : https://arxiv.org/abs/2105.14491
# code reference : https://github.com/tech-srl/how_attentive_are_gats/blob/main/gatv2_conv_DGL.py
class GraphAttentionLayerV2(nn.Module):
    def __init__(
        self, no_A, in_features, out_features, dropout, multi_head=4, ratio=16
    ):
        super(GraphAttentionLayerV2, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_squeeze = out_features // ratio
        self.no_A = no_A
        self.multi_head = multi_head

        self.W_src = nn.ModuleList()
        self.W_dst = nn.ModuleList()
        self.feat_drop = nn.ModuleList()
        self.A = nn.ModuleList()
        self.norm = nn.ModuleList()
        # self.norm_src = nn.ModuleList()
        # self.norm_dst = nn.ModuleList()

        for _ in range(no_A + 1):
            self.feat_drop.append(nn.Dropout(dropout))
            self.W_src.append(
                MakeParameterW(self.in_features, self.out_features_squeeze * multi_head)
            )
            self.W_dst.append(
                MakeParameterW(self.in_features, self.out_features_squeeze * multi_head)
            )
            self.A.append(MakeParameterAV2(self.multi_head, self.out_features_squeeze))
            self.norm.append(Norm(self.out_features_squeeze))
            # self.norm_src.append(Norm(self.out_features_squeeze * multi_head))
            # self.norm_dst.append(Norm(self.out_features_squeeze * multi_head))

        if in_features != out_features:
            self.map = nn.Linear(in_features, out_features, True)
        else:
            self.map = nn.Identity()

        self.squeeze = nn.Linear(
            self.out_features_squeeze * (no_A + 1), out_features, True
        )

    def forward(
        self,
        V,
        adj,
    ):

        n_node = V.size()[1]
        batch_size = V.size()[0]
        for num_A in range(self.no_A + 1):
            feat_src = feat_dst = self.feat_drop[num_A](V)

            feat_src = torch.matmul(feat_src, self.W_src[num_A]())
            # feat_src = self.norm_src[num_A](feat_src)
            feat_dst = torch.matmul(feat_dst, self.W_dst[num_A]())
            # feat_dst = self.norm_dst[num_A](feat_dst)

            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            e = (
                feat_src.repeat(1, 1, n_node).view(
                    batch_size,
                    n_node * n_node,
                    self.multi_head,
                    self.out_features_squeeze,
                )
                + feat_dst.repeat(1, n_node, 1).view(
                    batch_size, -1, self.multi_head, self.out_features_squeeze
                )
            ).view(
                batch_size, n_node, n_node, self.multi_head, self.out_features_squeeze
            )
            e = leaky_relu(e, inplace=True)

            # (batch_size, n_node, n_node, self.multi_head, features_squeeze) -> (batch_size, n_node, n_node, self.multi_head)
            e = (e * self.A[num_A]()).sum(dim=-1)

            if num_A < self.no_A:
                num_adj = adj[:, :, num_A, :]
            else:
                num_adj = torch.eye(n_node).to(adj.device)

            # == num_adj = torch.stack((num_adj, ) * self.multi_head, -1)
            ## batch_size, n_node, n_node, 1
            num_adj = num_adj.unsqueeze(-1)

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(num_adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout, training=self.training)

            feat_src = feat_src.reshape(
                batch_size, n_node * self.multi_head, self.out_features_squeeze
            )
            attention = attention.reshape(batch_size, n_node, n_node * self.multi_head)
            attention = torch.matmul(attention, feat_src)
            attention = self.norm[num_A](attention)

            if num_A == 0:
                output = attention
            else:
                output = torch.cat((output, attention), -1)

        output = self.squeeze(output)
        output += self.map(V)

        return output, adj

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class MakeDenseGAT(nn.Module):
    def __init__(
        self,
        input_feature,
        no_A,
        repeat_time,
        GraphAttentionLayer=GraphAttentionLayer,
        drop=0.3,
    ):
        super(MakeDenseGAT, self).__init__()
        self.layers = nn.ModuleList()
        for repeat in range(repeat_time):
            self.layers.append(
                GraphAttentionLayer(
                    no_A, input_feature * (repeat + 1), input_feature, drop
                )
            )
        self.squeeze_block = GraphAttentionLayer(
            no_A, input_feature * (repeat_time + 1), input_feature, drop
        )

    def forward(self, V, A):
        input_V = V.clone()
        for layer in self.layers:
            update_V = layer(input_V, A)[0]
            input_V = torch.cat((input_V, update_V), dim=-1)
        output = self.squeeze_block(input_V, A)[0]
        return output, A


class TuneSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class DiffPooling(nn.Module):
    def __init__(
        self,
        in_feature,
        out_feature,
        output_node,
        no_A=4,
        GraphAttentionLayer=GraphAttentionLayer,
        drop=0.3,
    ):
        """
        in_feature : input node features amount, equal V.shape[2] #V = (batch, node_amount, feature_amount)
        output_out_feature : represent to output node amount and output node feature.
        """
        super(DiffPooling, self).__init__()
        self.drop = drop
        self.no_A = no_A
        self.out_feature = out_feature
        self.output_node = output_node
        if output_node != 1:
            ratio = 16
        else:
            ratio = 1

        self.feature_layer = GraphAttentionLayer(
            no_A, in_feature, out_feature, drop, 4, ratio
        )
        self.adjacent_layer = GraphAttentionLayer(
            no_A, in_feature, output_node, drop, 4, ratio
        )

    def forward(self, X, A):
        # let's take 2708 nodes / 4 adj as input

        # (1,2708, input_feature) -> (1,2708, out_feature)
        X_feature = F.relu(self.feature_layer(X, A)[0])

        # (1,2708, input_feature) -> (1,2708, output_node) -> 2708 with softmax
        S = F.softmax(self.adjacent_layer(X, A)[0], dim=-1)
        # (1,2708, output_node) -> (1, output_node, 2708)
        S_T = torch.transpose(S, 1, 2)

        if self.output_node == 1:
            # (1, output_node, 2708)* (1,2708, out_feature) -> (1, output_node, out_feature)
            output = F.relu(torch.bmm(S_T, X_feature))
            output = output.reshape(-1, X.shape[2])
            return output, A
        else:
            # (1, output_node, 2708)*(1,2708, out_feature) -> (1, output_node, out_feature)
            X_feature = F.leaky_relu(torch.bmm(S_T, X_feature))
            # (1, output_node, 2708)*(1, 2708, 2708*4) -> (1, output_node, 2708*4)
            A_update = torch.bmm(S_T, A.reshape(-1, A.shape[1], self.no_A * A.shape[1]))
            # (1, output_node, 2708*4) -> (1, output_node*4, 2708)
            A_update = A_update.reshape(-1, A_update.shape[1] * self.no_A, A.shape[1])
            # (1, output_node*4, 2708)* (1,2708, output_node) -> (1, output_node*4, output_node)
            A_update = torch.bmm(A_update, S)
            # (1, output_node*4, output_node) -> (1, output_node, 4, output_node)
            A_update = A_update.reshape(
                -1, self.output_node, self.no_A, self.output_node
            )
            A_update = F.dropout(A_update, p=self.drop, training=self.training)

            return X_feature, A_update


class MakeParameterScale(nn.Module):
    def __init__(self):
        super(MakeParameterScale, self).__init__()
        self.parameter = nn.Parameter(torch.rand(1))

    def forward(self):
        return self.parameter


class GATV2(BaseNetwork):
    def __init__(
        self,
        input_feature,
        no_A=6,
        output_feature=128,
        class_=36,
        GraphAttentionLayer=GraphAttentionLayerV2,
    ):
        super(GATV2, self).__init__()
        self.fullflow = TuneSequential(
            GraphAttentionLayer(no_A, input_feature, 256, 0.3),
            MakeDenseGAT(256, no_A, 2, GraphAttentionLayer, 0.3),
            # DiffPooling(256, 256, 128, no_A, GraphAttentionLayer, 0.3),  # pooling
            GraphAttentionLayer(no_A, 256, 256, 0.3),
            # DiffPooling(256, 128, 64, no_A, GraphAttentionLayer, 0.1),  # pooling
            # GraphAttentionLayer(no_A, 256, 256, 0.1),
            # DiffPooling(128, 128, 1, no_A, GraphAttentionLayer, 0.1),  # pooling
        )

        self.mlp = nn.Sequential(nn.Linear(256, output_feature), nn.LeakyReLU(True))
        # self.embedding = nn.Linear(256, output_feature)
        self.class_output = nn.Linear(output_feature, class_)
        self.scale = MakeParameterScale()

    def l2_norm(self, input):
        """Perform l2 normalization operation on an input vector.
        code copied from liorshk's repository: https://github.com/liorshk/facenet_pytorch/blob/master/model.py
        """
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 2).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, inputs):
        V, A = inputs
        embedding, A = self.fullflow(V, A)
        embedding = self.mlp(embedding)
        # embedding = self.embedding(embedding)
        cross_ouptut = self.class_output(embedding)
        return cross_ouptut
