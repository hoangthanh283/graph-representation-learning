import json
from typing import Any

import numpy as np


class JsonHandler:
    """To read, dump file to json format. """

    def default(self, oo: np.int64) -> int:
        if isinstance(oo, np.int64):
            return int(oo)
        raise TypeError

    def read_json_file(self, filename: str) -> Any:
        with open(filename, encoding="utf-8-sig") as f:
            return json.load(f)

    def dump_to_file(self, data: Any, filename: str) -> None:
        with open(filename, "w", encoding="utf-8-sig") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
