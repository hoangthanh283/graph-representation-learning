import os
from typing import Any, Dict, List, Tuple, Union

from gnn.data_generator import augmentor, data_process
from gnn.data_generator.datasets.base_dataset import BaseDataset
from gnn.utils.json_handler import JsonHandler
from gnn.utils.logger.color_logger import color_logger


class DatapileDataset(BaseDataset):
    def __init__(self, data_config: Dict[str, Any], **kwargs: Dict[str, Any]):
        """Dataset loading from Datapile format.

        Args:
            config: Configuration parameters.
            data_type: Type of dataset (Train/Val/Test).
        """
        super(DatapileDataset, self).__init__()
        self.json_handler = JsonHandler()
        self.data_config = data_config
        data_type = kwargs.get("data_type")
        self.logger = color_logger(__name__, testing_mode=False)

        # Load dataset samples.
        self.list_samples = self._load_samples(kwargs.get("samples", None))
        self.num_samples = len(self.list_samples)

        # Load character set.
        self.charset = self._load_charset()
        self.char_to_id, self.id_to_char = self._map_charset_to_id(self.charset)

        # Load classes and key types.
        self.classes, self.key_types = self._load_classes()
        self.class_to_id, self.id_to_class = self._map_class_to_id(self.classes, self.key_types)

        # Loading all data processors.
        self.data_processors = self._load_data_processors()
        self.logger.info(f"Initialize {data_type} dataset, loading {self.num_samples} samples...")

    def _load_samples(self, samples: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Load data samples either directly from input or folder directories and format them.

        Args:
            samples: List of input samples, expected:
                [dict1, dict2, ...]
        Returns:
            list_samples: List of samples:
                [dict1, dict2, ...]
        """
        list_samples: List[Dict[str, Any]] = []
        if samples:
            list_samples = samples
        elif self.data_config.data_path:
            data_paths = self._load_sample_from_multi_folders(self.data_config.data_path)

            # Read image and label files.
            for label_path in data_paths:
                label = self.json_handler.read_json_file(label_path)
                list_samples.append(label)
        else:
            self.logger.error("Not found any dataset!")

        return list_samples

    def _load_sample_from_multi_folders(self, list_folders: List[str]) -> List[str]:
        """Load all samples from multiple data folders.

        Args:
            list_folders: List of data folders.

        Returns:
            all_sample: Lict of all samples items.
        """
        all_samples: List[str] = []
        for fp in list_folders:
            sample_list = self._load_sample_from_folder(fp)
            all_samples.extend(sample_list)

        return all_samples

    def _load_sample_from_folder(self, folder_path: str) -> List[str]:
        """Match all images and label files into dict's items.

        Args:
            folder_path (str): input folder for data which
            has data-pile format (contains image & labels folders)

        Return:
            all_samples (list): dict of matching image/label items
        """
        def _split_extention(fp: str) -> Tuple[str, str]:
            """Split file extention.
            Args:
                fp: File path name.

            Returns:
                File name, file extention.
            """
            filename, extention = os.path.splitext(fp)
            return (filename, extention)

        def _join_filepath(path: str, name: str, ext: str) -> str:
            """Join path, file, and extention into a file path.
            Args:
                path: Path name.
                name: File name.
                ext: File extention.

            Returns:
                File path name.
            """
            filepath = os.path.join(path, f"{name}{ext}")
            return filepath

        if not os.path.exists(folder_path):
            self.logger.warn(f"Found invalid data path: {folder_path}")
            return None

        label_samples: Dict[str, str] = {}
        for fp in os.listdir(folder_path):
            fn, ext = _split_extention(fp)
            label_samples[fn] = ext

        all_samples = [_join_filepath(folder_path, lbl_name, ext) for (lbl_name, ext) in label_samples.items()]
        return all_samples

    def _load_charset(self) -> List[str]:
        """Load charset from a given file path or meta data.

        Return:
            Character list.
        """
        # Load character set from file.
        charset_path = self.data_config.charset_path
        if charset_path:
            charset_item = self.json_handler.read_json_file(charset_path)
            charset = charset_item["charset"]
        else:
            self.logger.error("Not found any charset!")

        return charset

    @staticmethod
    def _map_charset_to_id(charset: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Map charset to idx.

        Args:
            charset: List of characters.

        Returns:
            A dict of char-idx items, a dict of idx-char items.
        """
        char_to_id = {char: idx for idx, char in enumerate(charset)}
        id_to_char = {idx: char for char, idx in char_to_id.items()}
        return (char_to_id, id_to_char)

    def _load_classes(self) -> Tuple[List[str], List[str]]:
        """Load class names from a given file path or previous meta data.

        Returns:
            A list of class names, a list of class types.
        """
        class_path = self.data_config.class_path
        if class_path:
            class_item = self.json_handler.read_json_file(class_path)
            class_list = class_item["classes"]
        else:
            self.logger.error("Not found class list!")

        key_types = self.data_config.key_types
        return class_list, key_types

    @staticmethod
    def _map_class_to_id(classes: List[str], key_types: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Map class to idx.

        Args:
            classes: A list of classes.
            key_types: A list of key types.

        Return:
            A dictionary of class to idx, a dictionary of idx to class.
        """
        class_to_id: Dict[str, int] = {}
        id_to_class: Dict[int, str] = {}
        for idx, lbl in enumerate(classes):
            class_to_id[lbl] = {}
            for k_id, kt in enumerate(key_types):

                # Add 1 value to the index --> 0 is back-ground (other textlines/entities).
                cls_idx = idx * len(key_types) + k_id + 1
                class_to_id[lbl][kt] = cls_idx
                id_to_class[cls_idx] = (lbl, kt)

        return class_to_id, id_to_class

    def _load_annotations(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Loading ground-truth in a given label file.

        Arguments:
            sample: Input sample dict.

        Return:
            polygons: List of polygon, each with [[x1, y1], [x2, y2], ..].
        """
        all_annotations: Dict[int, Any] = {}
        try:
            all_regions = sample["attributes"]["_via_img_metadata"]["regions"]
        except KeyError:
            for _, all_items in sample.items():
                all_regions = all_items["regions"]

        for idx, region in enumerate(all_regions):
            region_attr = region["region_attributes"]
            shape_attr = region["shape_attributes"]
            try:
                if shape_attr["name"] == "polygon":
                    all_x = shape_attr["all_points_x"]
                    all_y = shape_attr["all_points_y"]
                    polygon = list(zip(all_x, all_y))
                else:
                    x1 = shape_attr["x"]
                    y1 = shape_attr["y"]
                    x2 = shape_attr["width"] + x1
                    y2 = shape_attr["height"] + y1
                    polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

                annotation = {
                    "polygon": polygon,
                    "text": str(region_attr.get("label", "")),
                    "label": region_attr.get("formal_key", None),
                    "key_type": region_attr.get("key_type", None)
                }

            except KeyError as er:
                self.logger.error(er)
                continue

            if annotation["text"]:
                all_annotations[idx] = annotation
        return all_annotations

    def _load_data_processors(self) -> List[Union[augmentor.BaseAugmentor, data_process.BaseDataProcess]]:
        """Load all data processors (augmentations/preprocessing/..etc). """
        data_processors = []

        # Add augmentation methods.
        for aug in self.data_config.augmentations:
            aug_process = getattr(augmentor, aug)._from_config(self.data_config.augmentations[aug])
            self.logger.info(f"Type data processor: {aug_process.__class__.__name__}")
            data_processors.append(aug_process)

        # Add preprocessing methods.
        for pre in self.data_config.data_process:
            pre_process = getattr(data_process, pre)._from_config(self.data_config.data_process[pre])
            self.logger.info(f"Type data processor: {pre_process.__class__.__name__}")
            data_processors.append(pre_process)
        return data_processors

    def __getitem__(self, index: int, retry: int = 0) -> Dict[str, Any]:
        label = self._load_annotations(self.list_samples[index])
        sample = {
            "label": label,
            "charset": self.charset,
            "classes": self.classes,
            "char_to_id": self.char_to_id,
            "id_to_char": self.id_to_char,
            "class_to_id": self.class_to_id,
            "id_to_class": self.id_to_class,
        }
        for data_processor in self.data_processors:
            sample = data_processor(sample)
        return sample

    def __len__(self) -> int:
        return self.num_samples
