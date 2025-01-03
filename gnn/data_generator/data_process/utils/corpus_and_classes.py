import json
import os
from typing import List, Tuple

from gnn.data_generator.data_process.utils.normalize_text import normalize_text


def generate_corpus_and_classes(data_folder: str, list_none: List[str] = ["None", ""], data_format: str = "datapile"
                                ) -> Tuple[List[str], List[str]]:
    """Generate corpus and classes from DataPile Format Json

    Args:
        data_folder: Path to folder that contains json file
            of DataPile Format json
        list_none: List of classes represent None (Negative class)

    Return:
        corpus: (str) Character in corpus
        class_list (list): List of classes

    """
    corpus = ""
    class_list = []
    # Get data folder top level tree
    root, dirs, files = next(os.walk(data_folder))

    for fname in files:
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(root, fname)
        data = json.load(open(fpath, encoding="utf-8"))
        if data_format == "datapile":
            regions = data["attributes"]["_via_img_metadata"]["regions"]
            for region in regions:
                if region["region_attributes"]["formal_key"] not in list_none:
                    text = normalize_text(region["region_attributes"]["label"])
                    corpus += text
                    class_list.append(region["region_attributes"]["formal_key"])
        elif data_format == "athena":
            regions = data["regions"]
            for region in regions:
                if region["region_attributes"]["formal_key"] not in list_none:
                    text = (normalize_text(region["region_attributes"].get("text", "")
                                           or region["region_attributes"].get("label", "")))
                    corpus += text
                    class_list.append(region["region_attributes"]["formal_key"])
        else:
            raise NotImplementedError

    corpus = list(set(corpus))
    corpus.sort()
    corpus = "".join(corpus)

    class_list = list(set(class_list))
    class_list.sort()
    class_list.insert(0, "None")
    return corpus, class_list
