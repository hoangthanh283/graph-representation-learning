from typing import Any, Dict, List

import numpy as np

from gnn.data_generator.data_process.base_data_process import BaseDataProcess


class EdgeLabeling(BaseDataProcess):
    def __init__(self, is_directed: bool = False):
        """Labeling edge for each pair nodes represent by a adjacency matrix.
        With 1 indicates there is a connection, 0 means there is no connection.

        Args:
            is_directed: Whether the node link is directed or not.
        """
        super(EdgeLabeling, self).__init__()
        self.is_directed = is_directed

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(sample)

    def _make_link_label(self, entities: List[Dict[str, Any]], class_id: Dict[str, Any]) -> np.array:
        """Make numpy array indices of linking labels.

        Args:
            entities: List of entitiy types.
            class_id: A dictionary mapping label name & label idx.

        Returns:
            A numpy array of linking indices.
        """
        target_shape = (len(entities), len(entities))
        link_target = np.zeros(target_shape)
        for en in entities:
            for link in en["linking"]:

                # link = [[class_1, key_type_1], [class_2, key_type_2]].
                src = class_id[link[0][0]][link[0][1]]
                des = class_id[link[1][0]][link[1][1]]

                # Assign linking.
                link_target[src, des] = 1
                if not self.is_directed:
                    link_target[des, src] = 1

        return link_target

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
        link_label = self._make_link_label(textlines, class_to_id)
        sample["link_label"] = link_label
        return sample
