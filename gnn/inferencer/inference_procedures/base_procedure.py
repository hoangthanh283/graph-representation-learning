import os
from typing import Any, Dict, List, Tuple

import munch
import torch
import torch.nn as nn

from gnn.inferencer import post_processing
from gnn.inferencer.post_processing import PostProcessBase
from gnn.utils.checkpoint_handler import CheckpointHandler
from gnn.utils.logger.color_logger import color_logger


class BaseProcedure:
    def __init__(self, model: nn.Module, config: munch.munchify, **kwargs: Dict[str, Any]):
        """Base inference procedure.

        Args:
            model: A network instance.
            config: Configuration parameters.
        """
        self.logger = color_logger(__name__, testing_mode=False)
        self.logger.info("Initializing inference setups ...")

        # Make the infernce/output dir.
        self.config = config
        self.checkpointer = CheckpointHandler()
        self.inference_dir = self._make_infer_output_dir()

        # Find the devices to set.
        self.device, self.device_ids = self._prepare_device(self.config.inference_settings.num_gpus)

        # Load the model and its weight if has.
        self.model = self._load_prev_checkpoint(model)
        self.model = self.model.to(self.device)

        # Load all post-processing methods
        self.post_processors = self._load_post_processing(self.config.inference_settings.post_processing)
        self.logger.info("Successful initializing inference setups ...")

    @classmethod
    def _from_config(cls, model: nn.Module, config: munch.munchify, **kwargs: Dict[str, Any]) -> "BaseProcedure":
        return cls(model, config, **kwargs)

    def _prepare_device(self, n_gpu_use: int) -> Tuple[torch.device, List[int]]:
        """Setup GPU device if available.

        Args:
            n_gpu_use: Number of gpus to set.

        Return:
            device: Torch device instance.
            list_ids: List of gpu ids.
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There's no GPU available on this machine, the process will be performed on CPU.")
            n_gpu_use = 0

        if n_gpu_use > n_gpu:
            self.logger.warning(
                f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are available.")
            n_gpu_use = n_gpu

        device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
        list_ids = list(range(n_gpu_use))
        return (device, list_ids)

    def _to_parallelize(self, instance: nn.Module) -> nn.DataParallel:
        """ For distributed training with multi-GPUs.

        Args:
            instance: Input model.
            distributed: Whether use multi-gpu or not.
            local_rank: Num of GPUs.
        """
        if self.config.distributed:
            return torch.nn.parallel.DistributedDataParallel(
                instance, device_ids=[
                    self.device_ids], output_device=[
                    self.config.local_rank], find_unused_parameters=True)
        else:
            return torch.nn.DataParallel(instance)

    def _make_infer_output_dir(self) -> str:
        """Make output directory to save model/log../etc,
        based on experiment and model names.
        """
        output_dir = os.path.join(self.config.output_dir,
                                  self.config.inference_settings.output_dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def _load_prev_checkpoint(self, model: nn.Module) -> nn.Module:
        """Load model weight & previous configurations.

        Args:
            model: Model instance.

        Return:
            model: Loaded model instance.
        """
        pretrained_path = self.config.checkpoint_path
        if pretrained_path:
            self.logger.info("Restoring pretrained checkpoint ...")
            checkpoint_dict = self.checkpointer.restore_checkpoint(pretrained_path)
            model_state = checkpoint_dict.get("state_dict", None)
            previous_opt = checkpoint_dict.get("config", None)
            if (model_state and previous_opt):
                model.load_state_dict(model_state, strict=False)
                self.logger.info("Loading pretrained model success!")
        else:
            self.logger.info("Not found any pretrained model!")
        return model

    def _load_post_processing(self, process_config: munch.munchify) -> List[PostProcessBase]:
        """Load all post processors.

        Args:
            process_config: Post-processing config.

        Returns:
            Configured post processors.
        """
        post_processors: List[PostProcessBase] = []
        for post_process in process_config:
            post_process = getattr(post_processing, post_process)._from_config(process_config[post_process])
            self.logger.info(f"Type post processor: {post_process.__class__.__name__}")
            post_processors.append(post_process)

        return post_processors

    def _init_dataloaders(self):
        """Initializing dataset loader for input samples. """
        raise NotImplementedError

    def step_process(self):
        """Step processing of the dataset loader. """
        raise NotImplementedError

    def __call__(self):
        """Run prediction. """
        raise NotImplementedError
