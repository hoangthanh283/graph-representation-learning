from collections import defaultdict
from typing import Any, Dict, List, Tuple

import neptune.new as neptune
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

from gnn.data_generator.base_dataloader import BaseDataLoader
from gnn.trainer import losses
from gnn.trainer.training_procedures.kv_procedure import KVProcedure
from gnn.utils.metric_tracker import Dictlist


class JointTrainingProcedure(KVProcedure):
    def __init__(self, model: nn.Module, config: Dict[str, Any], tasks: List,
                 ems_exp: neptune.run.Run = None, **kwargs: Dict[str, Any]):
        """Optmizing process for KV task. """
        super(JointTrainingProcedure, self).__init__(model, config, ems_exp, **kwargs)
        self.tasks = tasks
        self.criterions = {
            "node_property": losses.MSELoss(),
            "edge_mask": losses.BinaryCrossEntropyLoss(),
            "pairwise_distance": losses.CrossEntropyLoss(),
            "pairwise_similarity": losses.MSELoss()
        }

    def _init_dataloaders(self) -> Tuple[BaseDataLoader, BaseDataLoader, Tuple[str]]:
        """ Initializing all train/val/test dataset loader.

        Return:
            train_dataloader: List of training data-loader instances.
            val_dataloader: List of validation/testing data-loader instances.
        """

        # Initialize dataloader/dataset base.
        dataloader = BaseDataLoader(self.config)

        # Training data-loaders.
        train_dataset = dataloader._load_dataset("training")
        train_dataloader = dataloader._get_dataloader(train_dataset)

        # Validation data-loaders.
        val_dataset = dataloader._load_dataset("validation")
        val_dataloader = dataloader._get_dataloader(val_dataset)

        # Get list of class names for both key and value.
        class_idx_names: List[Tuple[int, str]] = []
        for idx, names in train_dataset.id_to_class.items():
            class_idx_names.append((idx, "_".join(names)))
        _, class_names = zip(*sorted(class_idx_names, key=lambda kk: kk[0]))
        class_names = tuple(["other"] + list(class_names))  # As other class's index is 0.

        ssl_train_dataset = dataloader._load_dataset("ssl_training")
        ssl_train_dataloader = dataloader._get_dataloader(ssl_train_dataset)

        ssl_val_dataset = dataloader._load_dataset("ssl_validation")
        ssl_val_dataloader = dataloader._get_dataloader(ssl_val_dataset)

        return train_dataloader, val_dataloader, ssl_train_dataloader, ssl_val_dataloader, class_names

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
        test_elements = [self.config.datasets.node_label_padding_value,
                         self.config.datasets.other_class_index]
        selected_mask = np.isin(np_y_true, test_elements, invert=True)
        np_y_pred = np_y_pred[selected_mask]
        np_y_true = np_y_true[selected_mask]

        # Map index class to class names.
        pred_mapped_classes = [self.class_names[idx] for idx in np_y_pred.tolist()]
        lbl_mapped_classes = [self.class_names[idx] for idx in np_y_true.tolist()]

        # Get classification reports.
        score_matrix = classification_report(lbl_mapped_classes, pred_mapped_classes,
                                             output_dict=True, zero_division=0)

        # Set score items with the item name.
        score_items = score_matrix["macro avg"]
        score_items = {"{0}_{1}".format(item_name, kk): vv for (kk, vv) in score_items.items()}

        # Return predicted values and ground-truth values.
        input_items = {"pred": pred_mapped_classes, "lbl": lbl_mapped_classes}
        return score_items, input_items

    def _step_process(self, batch_sample: torch.Tensor, ssl_batch_sample: torch.Tensor,
                      **kwargs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """Step process for each batch sample.

        Args:
            batch_sample: Input batch samples.

        Return:
            loss: Loss value.
            metric_scores: Metric scores (f1 score, precision, recall, ... etc).
        """
        # Set list of batch inputs.
        total_loss = []

        batch_inputs = [
            batch_sample["textline_encoding"].float().to(self.device),
            batch_sample["adjacency_matrix"].float().to(self.device)
        ]
        # downstream task forward
        batch_targets = batch_sample["node_label"].to(self.device)
        batch_predicts = self.model.forward(batch_inputs)
        loss = self.criterion(batch_predicts, batch_targets)
        total_loss.append(loss)

        if ssl_batch_sample is not None:
            ssl_batch_inputs = [
                ssl_batch_sample["textline_encoding"].float().to(self.device),
                ssl_batch_sample["adjacency_matrix"].float().to(self.device)
            ]
            # auxiliary tasks forward
            for task in self.tasks:
                if task == "node_property":
                    ssl_batch_targets = ssl_batch_sample[task]
                    ssl_batch_targets = ssl_batch_targets.to(self.device)
                    ssl_batch_predicts = self.model.forward(
                        ssl_batch_inputs, task=task
                    )
                    loss = self.criterions[task](ssl_batch_predicts, ssl_batch_targets)
                    total_loss.append(loss)
                elif task in ["edge_mask", "pairwise_distance", "pairwise_similarity"]:
                    ssl_batch_edges = ssl_batch_sample[task + "_indices"]
                    ssl_batch_targets = ssl_batch_sample[task + "_targets"]
                    ssl_batch_targets = ssl_batch_targets.to(self.device)
                    ssl_batch_predicts = self.model.forward(ssl_batch_inputs, ssl_batch_edges, task=task)
                    loss = self.criterions[task](ssl_batch_predicts, ssl_batch_targets)
                    total_loss.append(loss)

        loss = sum(total_loss)
        # Calculate the loss.

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

    def _run_train_step(self, batch_sample: torch.Tensor, ssl_batch_sample: torch.Tensor,
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
            loss, metric_scores, input_items = self._step_process(batch_sample, ssl_batch_sample, **kwargs)
            loss.backward()

            # Clip the big gradient value, in the range of 0.0 to 1.
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optimize_settings.max_grad_norm)
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
            loss, metric_scores, input_items = self._step_process(batch_sample, None, **kwargs)

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
        for _ in tqdm(range(self.max_train_length), desc=f"Training - Epochs {epoch}"):
            try:
                train_batch = self.train_iter.next()
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                train_batch = self.train_iter.next()

            try:
                ssl_train_batch = self.ssl_train_iter.next()
            except StopIteration:
                self.ssl_train_iter = iter(self.ssl_train_loader)
                ssl_train_batch = self.ssl_train_iter.next()

            tr_metric_scores, tr_input_items = self._run_train_step(train_batch, ssl_train_batch)
            train_metrics._update(tr_metric_scores)
            self.tb_writer.add_scalar("Train step loss", tr_metric_scores["loss"], self.global_step)
            self.ems_exp["Train/step_loss"].log(tr_metric_scores["loss"], self.global_step)
            self.model.zero_grad()
            self.global_step += 1

        train_metrics = train_metrics._result()
        self.logger.info(f"Training epoch: {epoch} step: {self.global_step} metrics: {train_metrics}")
        for metric_name, score in train_metrics.items():
            self.tb_writer.add_scalar(f"Train {metric_name}", score, epoch)
            self.ems_exp[f"Train/{metric_name}"].log(score, epoch)

        # Run validation steps.
        val_metrics = Dictlist()
        val_item_dict = defaultdict(list)
        for val_batch in self.val_loader:
            val_metric_scores, val_input_items = self._run_val_step(val_batch)
            val_metrics._update(val_metric_scores)
            for kn, item_values in val_input_items.items():
                val_item_dict[kn].extend(item_values)

        val_metrics = val_metrics._result()
        self.logger.info(f"Validation metrics: {val_metrics}")
        for metric_name, score in val_metrics.items():
            self.tb_writer.add_scalar(f"Val {metric_name}", score, epoch)
            self.ems_exp[f"Validation/{metric_name}"].log(score, epoch)

        # Log classification report for all classes.
        val_report = classification_report(val_item_dict["lbl"], val_item_dict["pred"])
        self.logger.info("Classification report")
        self.logger.info(f"\n{val_report}")
        return val_metrics

    def __call__(self) -> str:
        """Optimizing process. """
        best_loss = float("inf")
        self.logger.info("Start optimizing ...")
        self.model.zero_grad()
        (
            self.train_loader,
            self.val_loader,
            self.ssl_train_loader,
            self.ssl_val_loader,
            self.class_names
        ) = self._init_dataloaders()
        self.max_train_length = max(len(self.train_loader), len(self.ssl_train_loader))
        self.train_iter = iter(self.train_loader)
        self.ssl_train_iter = iter(self.ssl_train_loader)

        for epoch in range(self.optimize_config.num_epochs):
            torch.cuda.empty_cache()
            metrics = self._optimize_per_epoch(epoch)
            self._update_learning_rate(epoch, self.global_step)

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
        return self.model_dir
