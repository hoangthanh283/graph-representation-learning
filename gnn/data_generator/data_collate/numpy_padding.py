from typing import Any, Dict, List, Tuple

import numpy as np

from gnn.data_generator.data_collate.base_collate import BaseCollate


class NumpyPadding(BaseCollate):
    def __init__(self, name_value_pairs: dict, only_selected_items: bool = False):
        """Normailze numpy inputs via getting the maximum size then padding the the rest.

        Args:
            name_value_pairs: Dict of names and their values to pad (eg: textline_encoding: 0, node_label: -100, ..etc).
            only_selected_items: Whether return output with only item names in input_names or not
        """
        super(NumpyPadding, self).__init__()
        self.name_value_pairs = name_value_pairs
        self.only_selected_items = only_selected_items

    def __call__(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Normalize item sizes.
        inputs = self._normalize_shape(inputs)

        # Whether to filter items based on names or not.
        if self.only_selected_items:
            inputs = self._return_selected_items(inputs)
        return inputs

    def _set_padding(self, padding_size: Tuple[int, int], pad_type: str = "symmetric") -> Tuple[int, int]:
        """Set the padding size based on the padding type.

        Args:
            padding_size: Initial padding size.
            pad_type: Type of padding (Edge/Symmetric).

        Return
            Padding size that set based on pad_type
        """
        if pad_type == "symmetric":
            padding_sizes = tuple([(s // 2, s - s // 2) for s in padding_size])
        elif pad_type == "edge":
            padding_sizes = tuple([(s, 0) for s in padding_size])
        else:
            raise ValueError(f"{pad_type} not found")
        return padding_sizes

    def _return_selected_items(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Only return items that are predefined in name value pairs.

        Args:
            inputs: List of input items.

        Return:
            Selected input items.
        """
        for idx in range(len(inputs)):
            try:
                inputs[idx] = {name: item for name, item in inputs[idx].items() if name in self.name_value_pairs.keys()}
            except Exception as er:
                raise ValueError(f"Error in filtering items: {er}")
        return inputs

    def _normalize_shape(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize shape of value in input items.

        Args:
            inputs: List of items.

        Return:
            Normalized items.
        """
        for (in_name, value) in self.name_value_pairs.items():

            # Get the inputs with a given name.
            max_size = 0
            _inputs = list(map(lambda k: k.get(in_name, None), inputs))
            _inputs = list(filter(None.__ne__, _inputs))

            if not _inputs:
                continue

            # Check if whether is numpy array type --> then
            # normalize their size according to the maximum size.
            if all([isinstance(kk, np.ndarray) for kk in _inputs]):
                list_shapes = list(map(lambda kk: list(kk.shape), _inputs))

                # Find the maximum shape.
                max_size = max(list_shapes, key=lambda kk: np.prod(np.array(kk)))

                # Get paddings for each input.
                pad_sizes = list(map(lambda kk: np.subtract(max_size, kk), list_shapes))

                # Set type for padding.
                pad_sizes = list(map(lambda kk: self._set_padding(kk, pad_type="symmetric"), pad_sizes))

                # Padding inputs with pad sizes.
                paded_input = list(map(lambda kk: np.pad(kk[0], kk[1], constant_values=value), zip(_inputs, pad_sizes)))

                # Update new value to the input items.
                for idx in range(len(inputs)):
                    inputs[idx][in_name] = paded_input[idx]

        return inputs
