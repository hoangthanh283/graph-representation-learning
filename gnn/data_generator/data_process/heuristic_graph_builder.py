from typing import Any, Dict, List

import numpy as np

from gnn.data_generator.data_process.base_data_process import BaseDataProcess
from gnn.data_generator.data_process.utils.graph_utils import Edge, Graph


class HeuristicGraphBuilder(BaseDataProcess):
    def __init__(self, num_edges: int, edge_type: str):
        """Constructing heuristic graphs via spatial relatons.

        Args:
            num_edges (int): number of edges (default: 6)
            edge_type (str): type of edges (default: normal_binary)
        """
        super(HeuristicGraphBuilder, self).__init__()
        self.num_edges = num_edges
        self.edge_type = edge_type

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(sample)

    def get_heuristic_adj_matrix(self, textlines: List[Dict[str, Any]], edge_type: str) -> List[Edge]:
        """Extract adjacency matrix.

        Args:
            textlines: List of textline items.
            edge_type: Type of edge.

        Returns:
            Adjacency matrix.
        """
        list_textlines = []
        for item in textlines:

            # Get cassia format location.
            location = item["polygon"]
            x1 = min(np.array(location)[:, 0])
            x2 = max(np.array(location)[:, 0])
            y1 = min(np.array(location)[:, 1])
            y2 = max(np.array(location)[:, 1])

            line_info = {
                "location": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "text": item["text"],
                "key_type": item.get("key_type", "other"),
                "type": item.get("label", "other")
            }
            list_textlines.append(line_info)

        # Construct heuristic graph.
        g = Graph(list_textlines, edge_type)
        return g.adj

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Take the inputs and create the graph. The output must have shape:
        N x E x N with N is number of nodes, E is number of edges.

        Args:
            sample: Sample item.

        Returns:
            Processed sample item.
        """
        all_lines = sample.get("label", None)
        if all_lines is None:
            return sample

        # Get order textlines by index.
        sample_items = sorted(sample["label"].items(), key=lambda k: k[0])
        ids, textlines = zip(*sample_items)

        # Construct adjacency matrix between textlines.
        adj_matrix = self.get_heuristic_adj_matrix(
            textlines,
            self.edge_type
        )
        n_vertex = len(textlines)

        # Remove rows and cols created by get_heuristic_graph_adj_mat.
        sample["adjacency_matrix"] = adj_matrix[:n_vertex, :, :n_vertex]
        return sample
