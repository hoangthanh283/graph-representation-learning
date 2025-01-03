from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from gnn.data_generator.data_process.base_data_process import BaseDataProcess
from gnn.data_generator.data_process.utils.normalize_text import normalize_text


class TextlineEncoding(BaseDataProcess):
    def __init__(self, is_normalized_text: bool):
        """Encode the bag-of-word and location of textlines into the numerical features.

        Args:
            is_normalized_text: Whether input textlines should be normalized or not.
        """
        super(TextlineEncoding, self).__init__()
        self.is_normalized_text = is_normalized_text

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(sample)

    def get_bow_matrix(self, textlines: List[str], char_to_idx: Dict[str, int]) -> np.array:
        """Get bag-of-word matrix.

        Args:
            textlines: A list of textlines.
            char_to_idx: A Dictionary of char-idx items.

        Returns:
            BOW matrix.
        """
        vectorizer = CountVectorizer(vocabulary=char_to_idx, analyzer="char", binary=True)

        # Whether is normalize text or not.
        all_texts = list(map(lambda line: normalize_text(str(line["text"]))
                             if self.is_normalized_text else str(line["text"]), textlines))

        # Tokenize and build vocab.
        bow_matrix = vectorizer.fit_transform(all_texts)
        bow_matrix = bow_matrix.toarray()
        return bow_matrix.astype(np.float32)

    def get_spatial_features_matrix(self, textlines: List[str]) -> np.array:
        """Encode sample's spatial information to (len(sample), 4).

        Args:
            textlines: A list of textlines.

        Returns:
            Coordinate spatial matrix.
        """
        def scale_non_zero(val: float, scale_val: float) -> float:
            """Set value to a non-zero value. """
            return float(val + scale_val) / (scale_val + 1.0)

        xs: List[float] = []
        ys: List[float] = []

        # Get corner points of textline"s locations.
        for textline in textlines:
            xs.extend([p[0] for p in textline["polygon"]])
            ys.extend([p[1] for p in textline["polygon"]])

        res: List[float] = []
        max_x, min_x = max(xs), min(xs)
        max_y, min_y = max(ys), min(ys)
        for textline in textlines:

            # Get x, y, width, height of current bounding box.
            feature = np.zeros(4, dtype=np.float)
            cur_xs = [p[0] for p in textline["polygon"]]
            cur_ys = [p[1] for p in textline["polygon"]]
            xx, yy = min(cur_xs), min(cur_ys)
            width, height = max(cur_xs) - xx, max(cur_ys) - yy

            # Scale/normalize locatoin to 0-1.
            feature[0] = scale_non_zero((xx - min_x) / (max_x - min_x), 0.1)
            feature[1] = scale_non_zero((yy - min_y) / (max_y - min_y), 0.1)
            feature[2] = scale_non_zero(width / (max_x - min_x), 0.1)
            feature[3] = scale_non_zero(height / (max_y - min_y), 0.1)
            res.append(feature)

        return np.array(res, dtype=np.float32)

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Take the inputs and process labels.

        Args:
            sample: Sample item.

        Returns:
            Processed sample item.
        """
        char_to_id = sample["char_to_id"]
        all_lines = sample.get("label", None)
        if all_lines is None:
            return sample

        # Get order textlines by index.
        sample_items = sorted(sample["label"].items(), key=lambda k: k[0])
        ids, textlines = zip(*sample_items)

        # Get text encoding.
        text_data = self.get_bow_matrix(textlines, char_to_id)

        # Get position encoding.
        spatial_data = self.get_spatial_features_matrix(textlines)

        # add text & position encodings.
        texline_ecodings = np.concatenate((text_data, spatial_data), axis=1)
        sample["textline_encoding"] = texline_ecodings
        return sample
