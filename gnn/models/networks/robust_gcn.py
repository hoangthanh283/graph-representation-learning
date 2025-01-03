import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init

from gnn.models.base_network import BaseNetwork


def make_linear_relu(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, num_edges, with_bias=True):
        super(GraphConv, self).__init__()
        self.C = output_dim
        self.L = num_edges
        self.F = input_dim
        self.gpu = torch.cuda.is_available()
        # h_weights: (L+1) (type of edges), F(input features), c(output dim)
        self.h_weights = nn.Parameter(
            torch.FloatTensor(self.F * (self.L + 1), self.C)
        )
        self.bias = (
            nn.Parameter(torch.FloatTensor(self.C)) if with_bias else None
        )
        # Todo: init the weight
        nn.init.xavier_normal_(self.h_weights)
        nn.init.normal_(self.bias, mean=0.0001, std=0.00005)

    def forward(self, V, A, preprocess_A=True):
        """
        Args:
            V: BxNxF
            A: BxNxNxL
            preprocess_A: Run preprocess adjacency matrix or not.
        """
        B = list(V.size())[0]
        N = list(V.size())[1]

        if preprocess_A:
            A = self.preprocess_adj(A)

        new_V = torch.matmul(A, V.view(-1, N, self.F)).view(
            -1, N, (self.L + 1) * self.F
        )

        # BN, (L+1)*F
        V_out = torch.matmul(new_V, self.h_weights) + self.bias.unsqueeze(0)
        return V_out.view(B, N, self.C)

    def preprocess_adj(self, adj):
        """ Preprocess adjacency matrix as proposed in paper """
        batch_size = list(adj.size())[0]
        num_nodes = list(adj.size())[1]

        identity_matrix = torch.unsqueeze(torch.eye(num_nodes), -1)
        identity = torch.stack([identity_matrix for _ in range(batch_size)])

        # Dirty way to get device
        cur_device = next(self.parameters()).device
        identity = identity.to(cur_device)

        adj = torch.cat([identity, adj], dim=-1)  # BxNxNx(L+1)
        adj = adj.view(batch_size * num_nodes, num_nodes, self.L + 1)  # (BN), N, (L+1)

        # Let's reverse stuffs a little bit for memory saving...
        # Since L is often much smaller than C, and we don't have that much mem
        # Aggregate node information first
        adj = adj.transpose(1, 2).reshape(-1, (self.L + 1) * num_nodes, num_nodes)  # BN(L+1), N
        return adj

    def __repr__(self):
        return f"GraphConv(in_dim={self.F}, out_dim={self.C}, num_edges={self.L}, bias={self.bias is not None})"


class NodeSelfAtten(nn.Module):
    def __init__(self, input_dim):
        super(NodeSelfAtten, self).__init__()
        self.F = input_dim
        self.f = make_linear_relu(input_dim, int(self.F // 8))
        self.g = make_linear_relu(input_dim, int(self.F // 8))
        self.h = make_linear_relu(input_dim, self.F)
        # Default tf softmax is -1, default torch softmax is flatten
        self.softmax = torch.nn.Softmax(-1)
        self.gamma = nn.Parameter(torch.FloatTensor(input_dim))
        torch.nn.init.normal_(self.gamma)

    def forward(self, V):
        f_out = self.f(V)  # B x N X F//8
        g_out = self.g(V).transpose(1, 2)  # B x F//8 x N
        h_out = self.h(V)  # B x N x F
        s = self.softmax(torch.matmul(f_out, g_out))  # B x N x N
        o = torch.matmul(s, h_out)
        return self.gamma * o + V

    def __repr__(self):
        return f"NodeSelfAttention(input_dim={self.F})"


# Copy from https://github.com/TachiChan/IJCAI2019_HGAT/blob/master/gat_layers.py
class GraphAttention(nn.Module):
    def __init__(self, f_in, f_out, n_head, attn_dropout=0.2, bias=True, slope=0.2):
        """ GAT layer for Heterogeneous Graph
        Args:
            f_in: Num feature input for model
            f_out: Num feature output for model
            n_head: Num adjency matrix
        """
        super(GraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=slope)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        """
        Args:
            h   (BxNxF)
            adj (BxNxNxL)
        """
        batch_size, n = h.size()[:2]
        # batch_size x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w)

        # batch_size x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)

        # batch_size x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)

        # batch_size x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(
            -1, -1, -1, n
        ).permute(0, 1, 3, 2)

        # batch_size x n_head x n x n
        attn = self.leaky_relu(attn)
        mask = 1 - adj.permute(0, 3, 1, 2)

        # batch_size x n_head x n x n
        attn.data.masked_fill_(mask > 0, float("-1e10"))
        attn = self.softmax(attn)

        # batch_size x n_head x n x n
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        # batch_size x n_head x n x f_out
        if self.bias is not None:
            output = output + self.bias

        output = output.permute(0, 2, 1, 3).reshape(
            batch_size, n, self.f_out * self.n_head
        )
        return output


class RobustGCN(BaseNetwork):
    def __init__(self, input_dim, output_dim, num_edges, net_size=256, use_attention=True):
        super(RobustGCN, self).__init__()
        self.output_dim = output_dim
        self.net_size = net_size
        self.emb1 = make_linear_relu(input_dim, self.net_size)
        self.dropout = nn.Dropout(p=0.5)
        self.gcn1 = GraphConv(self.net_size, self.net_size, num_edges)
        self.gcn2 = GraphConv(self.net_size, self.net_size, num_edges)
        self.gcn3 = GraphConv(self.net_size * 2, self.net_size, num_edges)
        half_net_size = self.net_size // 2
        self.emb2 = make_linear_relu(self.net_size * 2, half_net_size)
        self.use_attention = use_attention
        if use_attention:
            self.self_atten = NodeSelfAtten(half_net_size)

        self.gcn4 = GraphConv(half_net_size, half_net_size, num_edges)
        self.gcn5 = GraphConv(half_net_size, half_net_size, num_edges)
        self.classifier = nn.Linear(half_net_size, output_dim)

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
        g1 = F.relu(self.gcn1(embedding, A, preprocess_A))
        g1 = self.dropout(g1)

        # Second GraphConv
        g2 = F.relu(self.gcn2(g1, A, preprocess_A))
        g2 = self.dropout(g2)

        # Third GraphConv
        new_v = torch.cat([g2, g1], dim=-1)
        g3 = F.relu(self.gcn3(new_v, A, preprocess_A))
        g3 = self.dropout(g3)

        new_v = torch.cat([g3, g1], dim=-1)
        new_v = self.emb2(new_v)

        if self.use_attention:
            new_v = self.self_atten(new_v)
        new_v = self.dropout(new_v)

        # Final feature extractor
        g4 = F.relu(self.gcn4(new_v, A, preprocess_A))
        g4 = self.dropout(g4)
        g5 = F.relu(self.gcn5(g4, A, preprocess_A))
        return self.classifier(g5)
