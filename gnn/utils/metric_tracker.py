from typing import Any, Dict, List

import pandas as pd


class Dictlist(dict):
    """Handle multiple list keys-values. """

    def __setitem__(self, key: str, value: Any):
        try:
            self[key]
        except KeyError:
            super(Dictlist, self).__setitem__(key, [])
        self[key].append(value)

    def _update(self, dict_item: Dict[str, Any]) -> None:
        for (key, value) in dict_item.items():
            try:
                self[key]
            except KeyError:
                super(Dictlist, self).__setitem__(key, [])
            self[key].append(value)

    def _avg(self, key: str) -> float:
        list_value = self[key]
        return round(sum(list_value) / len(list_value), 3)

    def _result(self) -> Dict[str, Any]:
        return {key: self._avg(key) for key in self.keys()}


class MetricTracker:
    def __init__(self, *keys: Dict[str, Any], columns: List[str] = ["total", "counts", "average"], writer: Any = None):
        """This class tracks a metric during training for knowing whether the current value is the best so far. """
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=columns)
        self.reset()

    def _reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def _update(self, key: str, value: Any, nn: int = 1) -> None:
        if self.writer is not None:
            self.writer.add_scalar(key, value, nn)

        self._data.total[key] += value * nn
        self._data.counts[key] += nn
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def _avg(self, key: str) -> Dict[str, Any]:
        return self._data.average[key]

    def _result(self) -> Dict[str, Any]:
        return dict(self._data.average)
