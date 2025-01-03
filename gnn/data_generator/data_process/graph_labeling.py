from typing import Any, Dict

from gnn.data_generator.data_process.base_data_process import BaseDataProcess


class GraphLabeling(BaseDataProcess):
    def __init__(self):
        """Labeling index class of each node/entity (texline)."""
        super(GraphLabeling, self).__init__()

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(sample)

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Take the inputs and process labels.

        Args:
            sample: Sample item.

        Returns:
            Processed sample item.
        """
        all_lines = sample.get("label", None)
        if all_lines is None:
            return sample

        # Get order textlines by index.
        textlines = sorted(sample["label"].items(), key=lambda k: k[0])
        ids, textlines = zip(*textlines)
        class_to_id = sample["class_to_id"]

        # Make the labels.
        sample["graph_label"] = class_to_id[sample["graph_label"]]["value"]
        return sample
