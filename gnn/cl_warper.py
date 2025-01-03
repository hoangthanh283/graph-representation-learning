#!/usr/bin/env python
import os
import random
from typing import Any, Dict, List, Optional

import anyconfig
import munch
import numpy as np
import torch
import torch.nn as nn

from gnn.inferencer import inference_procedures
from gnn.trainer import training_procedures
from gnn.utils.checkpoint_handler import CheckpointHandler
from gnn.utils.constant import NEPTUNE_RUN
from gnn.utils.logger.color_logger import color_logger


class GNNLearningWarper:
    def __init__(self, model: nn.Module, config_path: Optional[str] = None, config: Optional[munch.munchify] = None):
        """An astract class for warping the process of graph learning.

        Args:
            model: A model instance to optimize.
            config_path: Path for config (yaml) file.
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load configuration from config file.
        assert config_path or config is not None
        self.config = self._from_config(config_path) if config_path else config

        # Seed the randomness for completely reproducible
        # results but not configimal for performance.
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        # Initialize output folders & logging dir.
        self.config.output_dir = self._make_output_dir()
        os.environ["OUTPUT_DIR"] = self.config.output_dir
        self.logger = color_logger(__name__, testing_mode=False)
        self.checkpointer = CheckpointHandler()

        if self.config.is_train:
            # Initialize training procedure.
            tr_procedure_type = self.config.procedure.type
            tr_procedure_args = self.config.procedure.args
            self.trainer = getattr(training_procedures, tr_procedure_type)._from_config(
                self.model, self.config, ems_exp=NEPTUNE_RUN, **tr_procedure_args)
        else:
            # Initialize inference procedure.
            infer_procedure_type = self.config.procedure.type
            infer_procedure_args = self.config.procedure.args
            self.inferencer = getattr(inference_procedures, infer_procedure_type)._from_config(
                self.model, self.config, **infer_procedure_args)

    @staticmethod
    def _from_config(config_path: str) -> munch.munchify:
        """Load and initialze configuration parameters.

        Args:
            config_path: Path to the configuration file.

        Returns:
            An instance of configuration parameters.
        """
        configs = anyconfig.load(config_path)
        munch_configs = munch.munchify(configs)
        if munch_configs.distributed:
            torch.cuda.set_device(munch_configs.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

        torch.backends.cudnn.benchmark = munch_configs.benchmark
        torch.backends.cudnn.deterministic = munch_configs.deterministic
        return munch_configs

    def _make_output_dir(self) -> str:
        """Make output directory to save model/log../etc based on experiment and model names. """
        output_dir = os.path.join(self.config.output_dir, self.config.experiment_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def _to_parallelize(self, instance: torch.nn, is_distributed: bool = False, num_rank: int = 0) -> \
            torch.nn.DataParallel:
        """For distributed training with multi-GPUs.

        Args:
            instance: Model instance to set parallel.
            is_distributed: Whether use multi-gpu or not.
            num_rank: Number of GPUs.

        Returns:
            Distributed model instance.
        """
        if is_distributed:
            instance = torch.nn.parallel.DistributedDataParallel(
                instance, device_ids=[num_rank], output_device=[num_rank], find_unused_parameters=True)
        else:
            instance = nn.DataParallel(instance)
        return instance

    def train(self) -> Any:
        """Optimizing model. """
        outputs = self.trainer()
        return outputs

    def predict(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Inferencing process. """
        outputs = self.inferencer(samples)
        return outputs
