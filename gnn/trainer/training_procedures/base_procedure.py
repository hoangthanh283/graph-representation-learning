import math
import os
from typing import Any, Dict, List, Tuple

import neptune.new as neptune
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from gnn.trainer import losses, lr_schedulers, optimizers
from gnn.utils.checkpoint_handler import CheckpointHandler
from gnn.utils.logger.color_logger import color_logger


class BaseProcedure:
    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 ems_exp: neptune = None, **kwargs: Dict[str, Any]):
        """Base training scheme for optimizing process.

        Args:
            model: A network instance.
            config: Configuration parameters.
        """
        self.logger = color_logger(__name__, testing_mode=True)
        self.logger.info("Initializing optimizing setups ...")

        # Make the save/output dir.
        self.config = config
        self.ems_exp = ems_exp
        self.model_dir = self._make_output_dir()
        self.checkpointer = CheckpointHandler()

        # Find the devices to set.
        self.device, self.device_ids = self._prepare_device(self.config.num_gpus)

        # Load the model and its weight if has.
        self.model = self._load_prev_checkpoint(model)
        self.model = self.model.to(self.device)

        # Initialize criterions, optimiers, learning rate schedulers based on config setups.
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer()
        self.lr_scheduler = self._init_lr_scheduler()

        # Setup visualization writer instance.
        sum_dir = os.path.join(self.config.output_dir, self.config.logging.summary_dir_name)
        self.tb_writer = SummaryWriter(sum_dir)
        self.logger.info("Successful initializing optimizing setups ...")

    @classmethod
    def _from_config(cls, model: nn.Module, config: Dict[str, Any], ems_exp: neptune = None,
                     **kwargs: Dict[str, Any]) -> "BaseProcedure":
        return cls(model, config, ems_exp, **kwargs)

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

    def _init_criterion(self) -> List[losses.BaseLoss]:
        """Initialize criterion from config. """
        try:
            loss_config = self.config.loss
            if loss_config.type:
                criterion = getattr(losses, loss_config.type)._from_config(loss_config.args)
                self.logger.info(f"Loss type: {criterion.__class__.__name__}")
                return criterion
            else:
                return None
        except Exception as er:
            self.logger.error(er)
            raise ValueError(er)

    def _init_lr_scheduler(self) -> List[lr_schedulers.BaseLearningRate]:
        """Initialize learning rate shcedualer from config. """
        lr_type = self.config.lr_scheduler.type
        lr_args = self.config.lr_scheduler.args
        try:
            if lr_type:
                lr_scheduler = getattr(lr_schedulers, lr_type)._from_config(lr_args)
                self.logger.info(f"Learning rate scheduler type: {lr_scheduler.__class__.__name__}")
                return lr_scheduler
            else:
                return None
        except Exception as error:
            self.logger.error("Error at %s", "division", exc_info=error)
            raise ValueError(error)

    def _init_optimizer(self) -> List[optimizers.BaseOptimizer]:
        """Initialize optimizer setting from configs. """
        try:
            type_optim = self.config.optimizer.type
            args_optim = self.config.optimizer.args
            if type_optim:
                optimizer = getattr(optimizers, type_optim)._from_config(args_optim)
                self.logger.info(f"Optimizer type: {optimizer.__class__.__name__}")
                optimizer = optimizer.get_optimizer(self.model.parameters())
                return optimizer
            else:
                return None
        except Exception as error:
            self.logger.error("Error at %s", "division", exc_info=error)
            raise ValueError(error)

    def _make_output_dir(self) -> str:
        """Make output directory to save model/log../etc,
        based on experiment and model names.
        """
        output_dir = os.path.join(self.config.output_dir, self.config.model_dir_name)
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

    def _update_learning_rate(self, epoch: int, step: int) -> float:
        """Update learning rate based on optimizer.

        Args:
            epoch: Current nth epoch.
            step: Current nth step.

        Return:
            lr: Updated learning rate.
        """
        lr = self.lr_scheduler._step_lr(epoch, step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def cosine_schedule_lambda(self, step: int, epoch: int, total_steps: int, base_value: float, max_value: float,
                               warmup_steps: int = 0) -> float:
        """Cyclical lambda scheduler with warmup and cosine annealing.

        Args:
            step: Current training step
            total_steps: Total number of training steps
            base_value: Minimum lambda value
            max_value: Maximum lambda value
            warmup_steps: Number of warmup steps with linear scaling
        """
        # Input validation
        step = max(0, min(step, total_steps))
        warmup_steps = min(warmup_steps, total_steps)

        # Calculate lambda
        if step < warmup_steps:
            # Linear warmup
            lambda_value = base_value + (max_value - base_value) * (step / warmup_steps)
        else:
            # Cosine annealing
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            lambda_value = base_value + 0.5 * (max_value - base_value) * (1 + math.cos(math.pi * progress))

        self.tb_writer.add_scalar("RP/Lambda", lambda_value, epoch)
        if self.ems_exp:
            self.ems_exp["RP/Lambda"].append(lambda_value)
        return lambda_value

    def _progress(self):
        """Visualizing the optimizing progress. """
        raise NotImplementedError

    def _init_dataloaders(self):
        """Initializing all train/val/test dataset loader. """
        raise NotImplementedError

    def __call__(self):
        """Optimizing process. """
        raise NotImplementedError
