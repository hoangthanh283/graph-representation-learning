import argparse

from gnn.cl_warper import GNNLearningWarper
from gnn.models import GraphCNNDropEdge

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CL configurations")
    parser.add_argument("--config", default=None, type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    """Input:
        The input to predict could be either:
            * (A/nested list of) la-ocr values with cassia format as below:
                [
                    {"location": [[0, 0], [0, 0], [0, 0], [0, 0]], "text": "abc"},
                    {"location": [[0, 0], [0, 0], [0, 0], [0, 0]], "text": "def"},
                    ...
                ]
            * (A/list of) la-ocr json file(s).

    Return:
        Predicted entities. Each output sample would have a format as below:
            [
                {"location": [[0, 0], [0, 0], [0, 0], [0, 0]], "text": "abc",
                    "key_type": "key", "confidence": 0.99, "formal_key": entity_name},

                {"location": [[0, 0], [0, 0], [0, 0], [0, 0]], "text": "edf",
                    "key_type": "value", "confidence": 0.99, "formal_key": entity_name},
                ...
            ]
    """
    model = GraphCNNDropEdge(input_dim=4369, output_dim=53, num_edges=6, net_size=256)
    warper = GNNLearningWarper(config_path=args.config)
    outputs = warper.predict("PUT_YOUR_INPUT_HERE")
