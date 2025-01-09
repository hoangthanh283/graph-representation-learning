from collections import defaultdict
from typing import Any, Dict, List, Tuple

import munch
import neptune.new as neptune
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

from gnn.data_generator.base_dataloader import BaseDataLoader
from gnn.trainer.training_procedures.base_procedure import BaseProcedure
from gnn.utils.metric_tracker import Dictlist


class KVProcedure(BaseProcedure):
    def __init__(self, model: nn.Module, config: munch.munchify,
                 neptune_exp: neptune = None, **kwargs: Dict[str, Any]):
        """Optmizing process for KV task. """
        super(KVProcedure, self).__init__(model, config, neptune_exp, **kwargs)
        self.global_step = 0
        self.config = config
        self.neptune_exp = neptune_exp
        self.activator = torch.nn.Softmax(dim=2)
        self.train_loader, self.val_loader, self.class_names = self._init_dataloaders()

    def _init_dataloaders(self) -> Tuple[BaseDataLoader, BaseDataLoader, Tuple[str]]:
        """ Initializing all train/val/test dataset loader.

        Return:
            train_dataloader: List of training data-loader instances.
            val_dataloader: List of validation/testing data-loader instances.
        """

        # Initialize dataloader/dataset base.
        dataloader = BaseDataLoader(self.config)

        # Training data-loaders.
        train_dataset = dataloader._load_dataset(self.config.data_config.dataset.type,
                                                 self.config.data_config.training,
                                                 data_type="training")
        train_dataloader = dataloader._get_dataloader(train_dataset, train_dataset.data_config)

        # Validation data-loaders.
        val_dataset = dataloader._load_dataset(self.config.data_config.dataset.type,
                                               self.config.data_config.validation,
                                               data_type="validation")
        val_dataloader = dataloader._get_dataloader(val_dataset, val_dataset.data_config)

        # Get list of class names for both key and value.
        class_idx_names: List[Tuple[int, str]] = []
        for idx, names in train_dataset.id_to_class.items():
            class_idx_names.append((idx, "_".join(names)))
        _, class_names = zip(*sorted(class_idx_names, key=lambda kk: kk[0]))
        class_names = tuple(["other"] + list(class_names))  # As other class"s index is 0.
        return train_dataloader, val_dataloader, class_names

    def _get_metric_scores(self, preds: torch.Tensor, gts: torch.Tensor,
                           item_name: str = "item") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get metric scores.
        Args:
            preds: Predicted tensors.
            gts: Ground truth tensors.
            item_name: Name of item in score output dict.
        Returns:
            Score output dict.
        """
        # Flatten predictions into numpy array.
        flatten_y_pred = torch.reshape(preds, shape=(-1,))
        np_y_pred = flatten_y_pred.cpu().detach().numpy()

        # Flatten groundtruth into numpy array.
        flatten_y_true = torch.reshape(gts, shape=(-1,))
        np_y_true = flatten_y_true.cpu().detach().numpy()

        # Ignore padding values, other class in label.
        test_elements = [self.config.data_config.dataset.args.node_label_padding_value,
                         self.config.data_config.dataset.args.other_class_index]
        selected_mask = np.isin(np_y_true, test_elements, invert=True)
        np_y_pred = np_y_pred[selected_mask]
        np_y_true = np_y_true[selected_mask]

        # Map index class to class names.
        pred_mapped_classes = [self.class_names[idx] for idx in np_y_pred.tolist()]
        lbl_mapped_classes = [self.class_names[idx] for idx in np_y_true.tolist()]

        # Get classification reports.
        try:
            score_matrix = classification_report(lbl_mapped_classes, pred_mapped_classes,
                                                 output_dict=True, zero_division=0)

            # Set score items with the item name.
            score_items = score_matrix["macro avg"]
        except Exception:
            score_items = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0.0}

        score_items = {"{0}_{1}".format(item_name, kk): vv for (kk, vv) in score_items.items()}

        # Return predicted values and ground-truth values.
        input_items = {"pred": pred_mapped_classes, "lbl": lbl_mapped_classes}
        return score_items, input_items

    def _step_process(self, batch_sample: torch.Tensor,
                      **kwargs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """Step process for each batch sample.
        Args:
            batch_sample: Input batch samples.
        Return:
            loss: Loss value.
            metric_scores: Metric scores (f1 score, precision, recall, ... etc).
        """
        # Set list of batch inputs.
        batch_inputs = [
            batch_sample["textline_encoding"].float().to(self.device),
            batch_sample["adjacency_matrix"].float().to(self.device)
        ]

        # Set list of batch targets.
        batch_targets = batch_sample["node_label"].to(self.device)

        # Run model prediction --> batch x nodes x classes.
        batch_predicts = self.model.forward(batch_inputs)

        # Calculate the loss.
        loss = self.criterion(batch_predicts, batch_targets)

        # Get the predicted class indexs,
        # batch x classes x nodes --> batch x nodes x classes.
        predicts = self.activator(batch_predicts)
        predicts = predicts.argmax(dim=-1)

        # Get metric scores.
        score_items, input_items = self._get_metric_scores(predicts, batch_targets,
                                                           item_name="Node classification")

        # Add metric scores.
        score_items.update({"loss": loss.item()})
        return loss, score_items, input_items

    def _run_train_step(self, batch_sample: torch.Tensor,
                        **kwargs: Dict[str, Any]) -> Tuple[Dict[int, Any], Dict[str, Any]]:
        """Run the batch training step and update the gradient/calculate metric scores.

        Args:
            batch_sample: Dict of input samples.

        Returns:
            Metric scores.
        """
        # Run predicting.
        self.model.train()
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            loss, metric_scores, input_items = self._step_process(batch_sample, **kwargs)
            loss.backward()

            # Clip the big gradient value, in the range of 0.0 to 1.
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        return metric_scores, input_items

    def _run_val_step(self, batch_sample: torch.Tensor,
                      **kwargs: Dict[str, Any]) -> Tuple[Dict[int, Any], Dict[str, Any]]:
        """Run the batch validation step and calculate metric scores.

        Args:
            batch_sample: Dict of input samples.
        """
        # Run predicting.
        self.model.eval()
        with torch.set_grad_enabled(False):
            _, metric_scores, input_items = self._step_process(batch_sample, **kwargs)

        return metric_scores, input_items

    def _optimize_per_epoch(self, epoch: int) -> Dict[int, Any]:
        """Run training step.

        Args:
            epoch: Ith epoch.

        Returns:
            val_metrics: Validation metric scores.
        """
        # Run training steps.
        train_metrics = Dictlist()
        training_bar = tqdm(self.train_loader, desc=f"Training - Epochs {epoch}")
        for train_batch in training_bar:
            tr_metric_scores, _ = self._run_train_step(train_batch)
            train_metrics._update(tr_metric_scores)
            self.tb_writer.add_scalar("Train_step_loss", tr_metric_scores["loss"], self.global_step)
            if self.neptune_exp:
                self.neptune_exp["Train/step_loss"].append(tr_metric_scores["loss"])
            training_bar.set_description("Epoch {0} - Train/Step loss: {1}".format(epoch, tr_metric_scores["loss"]))
            self.model.zero_grad()
            self.global_step += 1
            self.model.lambda_value = self.cosine_schedule_lambda(self.global_step, epoch,
                                                                  self.config.num_epochs * len(self.train_loader),
                                                                  base_value=1e-4, max_value=1.0,
                                                                  warmup_steps=5 * len(self.train_loader))

        train_metrics = train_metrics._result()
        self.logger.info(f"Training epoch: {epoch} step: {self.global_step} metrics: {train_metrics}")
        for metric_name, score in train_metrics.items():
            self.tb_writer.add_scalar(f"Train {metric_name}", score, epoch)
            if self.neptune_exp:
                self.neptune_exp[f"Train/{metric_name}"].append(score)

        # Run validation steps.
        val_metrics = Dictlist()
        val_item_dict = defaultdict(list)
        validation_bar = tqdm(self.val_loader, desc=f"Validation - Epochs {epoch}")
        for val_batch in validation_bar:
            val_metric_scores, val_input_items = self._run_val_step(val_batch)
            val_metrics._update(val_metric_scores)
            validation_bar.set_description("Epoch {0} - Val/Step loss: {1}".format(epoch, val_metric_scores["loss"]))
            for kn, item_values in val_input_items.items():
                val_item_dict[kn].extend(item_values)

        val_metrics = val_metrics._result()
        self.logger.info(f"Validation metrics: {val_metrics}")
        for metric_name, score in val_metrics.items():
            self.tb_writer.add_scalar(f"Val {metric_name}", score, epoch)
            if self.neptune_exp:
                self.neptune_exp[f"Validation/{metric_name}"].append(score)

        # Log classification report for all classes.
        val_report = classification_report(val_item_dict["lbl"], val_item_dict["pred"])
        val_report_dict = classification_report(val_item_dict["lbl"], val_item_dict["pred"], output_dict=True)
        macro_avg_val = val_report_dict["macro avg"]
        for macro_metric_name, macro_score in macro_avg_val.items():
            self.tb_writer.add_scalar(f"Macro Val {macro_metric_name}", macro_score, epoch)
            if self.neptune_exp:
                self.neptune_exp[f"Macro Validation/{macro_metric_name}"].append(macro_score)

        self.logger.info("Classification report")
        self.logger.info(f"\n{val_report}")
        # return val_metrics
        macro_avg_val["loss"] = val_metrics["loss"]
        return macro_avg_val

    def __call__(self) -> str:
        """Optimizing process. """
        best_loss = float("inf")
        self.logger.info("Start optimizing ...")
        self.model.zero_grad()
        for epoch in range(self.config.num_epochs):
            torch.cuda.empty_cache()
            metrics = self._optimize_per_epoch(epoch)
            self._update_learning_rate(epoch, self.global_step)
            # self.model.lambda_value = self.schedule_lambda(self.model.lambda_value, epoch, self.config.num_epochs)

            # Update model parameter distribution.
            for name, param in self.model.named_parameters():
                self.tb_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            # Save the model dict if get a lower loss value.
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                checkpoints = {
                    "epoch": epoch,
                    "config": dict(self.config),
                    "meta_data": metrics,
                    "state_dict": self.model.state_dict(),
                }
                self.checkpointer.save_checkpoint(checkpoints, self.model_dir)

        self.logger.info("Finish optimizing!")
        self.tb_writer.close()
        # To visualize representation space.
        # self.visualize_representation_space(self.val_loader)
        # return self.model_dir
        return metrics["f1-score"]
