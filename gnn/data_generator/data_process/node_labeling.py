from typing import Any, Dict, List

import numpy as np

from gnn.data_generator.data_process.base_data_process import BaseDataProcess


class NodeLabeling(BaseDataProcess):
    def __init__(self):
        """Labeling index class of each node/entity (texline). """
        super(NodeLabeling, self).__init__()

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(sample)

    def _make_node_label(self, entities: List[Dict[str, Any]], class_id: Dict[str, Any]) -> np.array:
        """Make numpy array indices of node labels.

        Args:
            entities: List of entitiy types.
            class_id: A dictionary mapping label name & label idx.

        Returns:
            A numpy array of node label indices.
        """
        node_targets = [class_id.get(en["label"], {}).get(en["key_type"], 0) for en in entities]
        node_targets = np.array(node_targets)
        return node_targets

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
        sample_items = sorted(sample["label"].items(), key=lambda k: k[0])
        ids, textlines = zip(*sample_items)
        class_to_id = sample["class_to_id"]

        # Make the labels.
        node_label = self._make_node_label(textlines, class_to_id)
        sample["node_label"] = node_label
        return sample
