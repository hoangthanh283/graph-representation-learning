import torch
from torch import nn
from torch.nn import functional as F

from gnn.models.networks.drop_robust_gcn import RobustFilterGraphCNNDropEdge


class SSLGCN(RobustFilterGraphCNNDropEdge):
    """The Core of Graph KV Module created by Ethan
    Using the original name as tribute to him and Marc, Dini
    who is the original writer of this source code
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_edges,
        n_pairwise_distance=4,
        n_graph_classes=204,
        net_size=256,
        use_attention=True,
    ):
        super(SSLGCN, self).__init__(
            input_dim,
            output_dim,
            num_edges,
            net_size=net_size,
            use_attention=use_attention,
        )
        half_net_size = self.net_size // 2

        # SSL output layers
        self.ssl_layers = nn.ModuleDict(
            {
                "node_property": nn.Linear(half_net_size, 1),
                "edge_mask": nn.Linear(half_net_size, 1),
                "pairwise_distance": nn.Linear(half_net_size, n_pairwise_distance),
                "pairwise_similarity": nn.Linear(half_net_size, 1),
                "graph_edit_distance": nn.Linear(net_size, 1),
                "graph_classification": nn.Linear(net_size, n_graph_classes),
            }
        )
    
    def get_node_emb(self, inputs, efficient_mode=True):
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
        new_v = torch.cat([g1, g2], dim=-1)
        g3 = F.relu(self.gcn3(new_v, A, preprocess_A))
        g3 = self.dropout(g3)

        new_v = torch.cat([g1, g3], dim=-1)
        new_v = self.emb2(new_v)

        if self.use_attention:
            new_v = self.self_atten(new_v)
        new_v = self.dropout(new_v)

        return new_v

    def forward(self, inputs, edges=None, task=None, efficient_mode=True):
        if task == "node_property":
            node_emb = self.get_node_emb(inputs, efficient_mode)
            return self.ssl_layers["node_property"](node_emb)
        elif task in ["edge_mask", "pairwise_distance", "pairwise_similarity"]:
            node_emb = self.get_node_emb(inputs, efficient_mode)
            node_emb = node_emb.view(-1, node_emb.shape[-1])
            src_emb = node_emb[edges[:, :, 0].view(-1)]
            dest_emb = node_emb[edges[:, :, 1].view(-1)]
            out = self.ssl_layers[task](torch.abs(src_emb - dest_emb))

            return out.view(
                edges.shape[0], edges.shape[1], out.shape[-1]
            )  # B x N x out_dim
        elif task in ["graph_edit_distance"]:
            src_node_emb = self.get_node_emb(
                inputs[:2], efficient_mode
            )  # B x N x half_net_size
            dest_node_emb = self.get_node_emb(
                inputs[2:], efficient_mode
            )  # B x N x half_net_size

            src_graph_emb = torch.cat(
                [
                    torch.max(src_node_emb, dim=1, keepdim=True)[0],
                    torch.mean(src_node_emb, dim=1, keepdim=True),
                ],
                dim=-1,
            )  # B x 1 x (8 * hidden_dim)
            dest_graph_emb = torch.cat(
                [
                    torch.max(dest_node_emb, dim=1, keepdim=True)[0],
                    torch.mean(dest_node_emb, dim=1, keepdim=True),
                ],
                dim=-1,
            )  # B x 1 x (8 * hidden_dim)

            out = self.ssl_layers["graph_edit_distance"](
                torch.abs(src_graph_emb - dest_graph_emb)
            )  # B x 1 x 1

            return out
        elif task in ["graph_classification"]:
            node_emb = self.get_node_emb(inputs, efficient_mode)
            graph_emb = torch.cat(
                [
                    torch.max(node_emb, dim=1, keepdim=True)[0],
                    torch.mean(node_emb, dim=1, keepdim=True),
                ],
                dim=-1,
            )  # B x 1 x net_size
            out = self.ssl_layers["graph_classification"](
                graph_emb
            )  # B x 1 x n_classes

            return out
        elif task in ['dgi']:
            node_emb = self.get_node_emb(
                inputs[:2], efficient_mode
            )  # B x N x half_net_size
            negative_node_emb = self.get_node_emb(
                inputs[2:], efficient_mode
            )  # B x N x half_net_size

            return node_emb, negative_node_emb

        # node classification
        node_emb = super(SSLGCN, self).get_node_emb(inputs, efficient_mode)
        return self.classifier(node_emb)
