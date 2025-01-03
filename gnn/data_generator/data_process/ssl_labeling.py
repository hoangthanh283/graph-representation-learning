from typing import Any, Dict, List

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from gnn.data_generator.data_process.base_data_process import BaseDataProcess


class SSLLabeling(BaseDataProcess):
    def __init__(self, tasks: List, is_directed: bool = False):
        """Labeling index class of each node/entity (texline).

        Args:
            task_type: Type of prediction task (node_classification/link_prediction)
            is_directed: Whether the node link is directed or not.
        """
        super(SSLLabeling, self).__init__()
        self.tasks = tasks
        self.is_directed = is_directed

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(sample)

    def _calculate_node_property(
        self, vertex: np.array, adj_matrix: np.array, property: str = "degree"
    ) -> np.array:
        sum_adj_matrix = np.sum(adj_matrix, axis=1)  # shape N, N
        assert (sum_adj_matrix <= 1).all()

        return np.sum(sum_adj_matrix > 0, axis=1)  # shape N,

    def _calculate_edge_mask(
        self, vertex: np.array, adj_matrix: np.array, k: int
    ) -> np.array:
        sum_adj_matrix = np.sum(adj_matrix, axis=1)  # shape N, N
        assert (sum_adj_matrix <= 1).all()

        positive_edges = np.vstack(np.nonzero(sum_adj_matrix > 0))
        # TODO: retrieve a fixed number or dynamic k for positive/negative
        perm = np.random.permutation(positive_edges.shape[1])[:k]
        positive_edges = positive_edges[:, perm]  # shape 2 x k

        negative_edges = np.vstack(np.nonzero(sum_adj_matrix == 0))
        perm = np.random.permutation(negative_edges.shape[1])[:k]
        negative_edges = negative_edges[:, perm]  # shape 2 x k

        edges = np.concatenate([positive_edges, negative_edges], axis=1).reshape(
            (-1, 2)
        )  # shape 2 * k, 2
        edge_targets = np.concatenate([np.ones(k), np.zeros(k)], axis=0)  # shape 2 * k

        return edges, edge_targets

    def _calculate_pairwise_distance(
        self,
        vertex: np.array,
        adj_matrix: np.array,
        max_distance: int,
        k: int,
    ) -> np.array:
        # TODO: calculate shortest path length
        sum_adj_matrix = np.sum(adj_matrix, axis=1)  # shape N, N
        assert (sum_adj_matrix <= 1).all()

        graph = nx.convert_matrix.from_numpy_matrix(
            sum_adj_matrix, parallel_edges=True, create_using=nx.MultiDiGraph
        )
        lengths = dict(
            nx.all_pairs_shortest_path_length(graph, cutoff=max_distance - 1)
        )
        distance = -np.ones((len(graph), len(graph))).astype(int)

        for u, p in lengths.items():
            for v, d in p.items():
                distance[u][v] = d

        distance[
            distance == -1
        ] = max_distance  # node pairs with no connection are treated as same as >= max_distance (out of range)
        distance = np.triu(
            distance
        )  # ignore duplicate connection. by default, self edge is 0-distance
        distance = (
            distance - 1
        )  # -1 is ignored (lower triangle), classes start from 0 to max_distance - 1. max_distance - 1 is for out of range pair.

        edges = np.vstack(np.nonzero(distance > -1))
        perm = np.random.permutation(edges.shape[1])[:k]

        edges = edges[:, perm].transpose()  # shape k, 2
        edge_targets = [
            distance[edges[i][0], edges[i][1]] for i in range(edges.shape[0])
        ]
        edge_targets = np.array(edge_targets)

        return edges, edge_targets  # k classes start from 0

    def _calculate_pairwise_attribute_similarity(
        self, vertex: np.array, adj_matrix: np.array, k: int
    ) -> np.array:

        similarity_matrix = cosine_similarity(vertex)  # N, N
        edges, edge_targets = [], []
        top_k_similar = np.argpartition(similarity_matrix, -k, axis=1)[:, -k:]
        for src in range(top_k_similar.shape[0]):
            for dest in top_k_similar[src].tolist():
                edges.append([src, dest])
                edge_targets.append(similarity_matrix[src, dest])

        top_k_dissimilar = np.argpartition(similarity_matrix, k, axis=1)[:, :k]
        for src in range(top_k_dissimilar.shape[0]):
            for dest in top_k_dissimilar[src].tolist():
                edges.append([src, dest])
                edge_targets.append(similarity_matrix[src, dest])

        edges = np.array(edges)  # shape 2 * N * k, 2
        edge_targets = np.array(edge_targets)  # shape 2 * N * k,

        return edges, edge_targets

    def _calculate_graph_edit_distance(self, src_adj_matrix: np.ndarray, dest_adj_matrix: np.ndarray,
                                       graph_edit_history: np.ndarray) -> np.ndarray:
        N, F, _ = src_adj_matrix.shape
        reconstructed_dest_adj_matrix = dest_adj_matrix.copy()
        graph_edit_history = sorted(graph_edit_history, key=lambda x: x[0])  # Sort according to the node-idx.
        for node, op in graph_edit_history:
            if op == "delete":
                reconstructed_dest_adj_matrix = np.insert(
                    reconstructed_dest_adj_matrix, node, np.zeros(F), axis=2
                )  # insert a new col
                reconstructed_dest_adj_matrix = np.insert(
                    reconstructed_dest_adj_matrix, node, np.zeros((1, F, 1)), axis=0
                )  # insert a new row

        node_cost = len(graph_edit_history)  # each operation cost 1
        edge_edit_delete_cost = np.sum(
            np.abs(reconstructed_dest_adj_matrix[:N, :, :N] - src_adj_matrix)
        )  # same shape
        edge_add_cost = (
            np.sum(reconstructed_dest_adj_matrix[N:, :, :])
            + np.sum(reconstructed_dest_adj_matrix[:, :, N:])
            - np.sum(reconstructed_dest_adj_matrix[N:, :, N:])
        )

        return node_cost + edge_edit_delete_cost + edge_add_cost

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Take the inputs and process labels.

        Args:
            sample: Sample item.

        Returns:
            Processed sample item.
        """

        vertex = sample["textline_encoding"]  # shape N x F
        adj_matrix = sample["adjacency_matrix"]  # shape N x n_edges x N
        for task in self.tasks:
            if task == "node_property":
                sample["node_property"] = self._calculate_node_property(
                    vertex, adj_matrix
                )
            elif task == "edge_mask":
                (
                    sample["edge_mask_indices"],
                    sample["edge_mask_targets"],
                ) = self._calculate_edge_mask(
                    vertex, adj_matrix, k=vertex.shape[0] // 10
                )
            elif task == "pairwise_distance":
                (
                    sample["pairwise_distance_indices"],
                    sample["pairwise_distance_targets"],
                ) = self._calculate_pairwise_distance(
                    vertex, adj_matrix, max_distance=4, k=vertex.shape[0] // 5
                )
            elif task == "pairwise_similarity":
                (
                    sample["pairwise_similarity_indices"],
                    sample["pairwise_similarity_targets"],
                ) = self._calculate_pairwise_attribute_similarity(
                    vertex, adj_matrix, k=3
                )
            elif task == "graph_edit_distance":
                aug_adj_matrix = sample["aug_adjacency_matrix"]
                graph_edit_history = sample["graph_edit_history"]
                sample["graph_edit_distance"] = self._calculate_graph_edit_distance(
                    adj_matrix, aug_adj_matrix, graph_edit_history
                )
            elif task == "dgi":
                negative_vertex = sample["negative_textline_encoding"]
                sample["dgi"] = np.concatenate([np.ones(vertex.shape[0]), np.zeros(negative_vertex.shape[0])], axis=0)

        return sample
