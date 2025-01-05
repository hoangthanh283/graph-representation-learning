from typing import Any, Callable, Dict, List

import anyconfig
import munch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler

from gnn.data_generator import data_collate, datasets
from gnn.data_generator.data_collate.base_collate import BaseCollate
from gnn.data_generator.datasets import BaseDataset
from gnn.utils.logger.color_logger import color_logger


class BaseDataLoader(DataLoader):
    def __init__(self, config: munch.munchify):
        """Data loader to loader batch samples for training process.

        Arguments:
            config: Predefined parameters.
        """
        self.config = config
        self.logger = color_logger(__name__, testing_mode=False)
        self.logger.info(f"Run {self.__class__.__name__}")

    @classmethod
    def _from_config(cls, config_path: str) -> Dict[str, Any]:
        """Initialze all configurations.

        Args:
            config_path: Path to the yml config file.

        Return:
            munch_configs: Config instance.
        """
        configs = anyconfig.load(config_path)
        munch_configs = munch.munchify(configs)
        return cls(munch_configs)

    def _load_collate_processors(self, collate_config: munch.munchify) -> List[BaseCollate]:
        """Load all data collate processors.

        Args:
            collate_config: Data collate config.

        Returns:
            Configured colldate processors.
        """
        collate_processors: List[BaseCollate] = []
        for collate in collate_config:
            collate_process = getattr(data_collate, collate)._from_config(collate_config[collate])
            self.logger.info(f"Type data collate processor: {collate_process.__class__.__name__}")
            collate_processors.append(collate_process)

        # Add default collate for collating lists of samples into batches.
        collate_processors.append(default_collate)
        return collate_processors

    def _load_dataset(self, dataset_type: str, args: Dict[str, Any], **kwargs: Dict[str, Any]) -> DataLoader:
        """Load dataset with a given configs.

        Args:
            data_type: Type of dataset (training/validation/testing/..etc).
            args: Arguments for that dataset.

        Return:
            data_loader: Instance of dataloader.
        """
        # Loading dataset.
        dataset = getattr(datasets, dataset_type)._from_config(args, **kwargs)
        return dataset

    def _get_dataloader(self, dataset: BaseDataset, data_config: munch.munchify, custom_collate: Callable = None,
                        **kwargs: Dict[str, Any]) -> DataLoader:
        """Load dataset with a given configs.

        Args:
            dataset: Dataset instance to make data loader.
            data_config: Data Configuration parameters for the current dataset.
            custom_collate: Customized collate function.

        Return:
            data_loader: Instance of dataloader.
        """
        try:
            # Set up data collates.
            if data_config.data_collate:
                data_collates = self._load_collate_processors(data_config.data_collate)
                data_collates = torchvision.transforms.Compose(data_collates)
            else:
                data_collates = None
            is_distributed = self.config.distributed
            if is_distributed:
                data_config.batch_size = data_config.batch_size // self.config.num_gpus
                data_sampler = DistributedSampler(dataset,
                                                  num_replicas=self.config.num_gpus,
                                                  rank=self.config.local_rank)

            # Loading dataloader.
            data_loader = DataLoader(
                dataset,
                batch_size=data_config.batch_size,
                num_workers=data_config.num_workers,
                drop_last=data_config.drop_last,
                pin_memory=data_config.pin_memory,
                collate_fn=custom_collate if custom_collate is not None else data_collates,
                shuffle=data_config.shuffle if is_distributed else False,
                sampler=data_sampler if is_distributed else None
            )
            self.logger.info("Initializing dataloader ...")
            return data_loader
        except Exception as er:
            self.logger.error(er)
            raise ValueError(er)
