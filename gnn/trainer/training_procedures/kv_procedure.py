import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from tqdm import tqdm

from gnn.data_generator.base_dataloader import BaseDataLoader
from gnn.trainer.training_procedures.base_procedure import BaseProcedure
from gnn.utils.metric_tracker import Dictlist


class KVProcedure(BaseProcedure):
    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 ems_exp: neptune = None, **kwargs: Dict[str, Any]):
        """Optmizing process for KV task. """
        super(KVProcedure, self).__init__(model, config, ems_exp, **kwargs)
        self.global_step = 0
        self.config = config
        self.ems_exp = ems_exp
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

        # Run model prediction -- batch x nodes x classes.
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
            if self.ems_exp:
                self.ems_exp["Train/step_loss"].append(tr_metric_scores["loss"])
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
            if self.ems_exp:
                self.ems_exp[f"Train/{metric_name}"].append(score)

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
            if self.ems_exp:
                self.ems_exp[f"Validation/{metric_name}"].append(score)

        # Log classification report for all classes.
        val_report = classification_report(val_item_dict["lbl"], val_item_dict["pred"])
        val_report_dict = classification_report(val_item_dict["lbl"], val_item_dict["pred"], output_dict=True)
        macro_avg_val = val_report_dict["macro avg"]
        for macro_metric_name, macro_score in macro_avg_val.items():
            self.tb_writer.add_scalar(f"Macro Val {macro_metric_name}", macro_score, epoch)
            if self.ems_exp:
                self.ems_exp[f"Macro Validation/{macro_metric_name}"].append(macro_score)

        self.logger.info("Classification report")
        self.logger.info(f"\n{val_report}")
        # return val_metrics
        macro_avg_val["loss"] = val_metrics["loss"]
        return macro_avg_val

    def schedule_lambda(self, init_value: int, epoch: int, num_epoches: int, factor: float = 0.9) -> float:
        rate = np.power(1.0 - epoch / float(num_epoches + 1), factor)
        new_value = init_value * rate
        self.tb_writer.add_scalar("RP/Lambda", new_value, epoch)
        if self.ems_exp:
            self.ems_exp["RP/Lambda"].append(new_value)
        return new_value

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

    def visualize_representation_space(self, dataloader: BaseDataLoader):
        """
        Visualize the representation space in 2D using t-SNE.

        Args:
            dataloader: Dataloader for the dataset.
        """
        self.model.eval()
        self.model.to(self.device)
        representations: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        validation_bar = tqdm(dataloader, desc="Visualize representation space ...")
        # Collect representations and labels.
        with torch.no_grad():
            for batch in validation_bar:
                # Set list of batch inputs.
                inputs = [
                    batch["textline_encoding"].float().to(self.device),
                    batch["adjacency_matrix"].float().to(self.device)
                ]
                # Set list of batch targets.
                targets = batch["node_label"].to(self.device)

                # Forward pass through the model to get representations.
                V, A = inputs
                A = A.permute(0, 1, 3, 2)
                try:
                    A = self.model.gcn1.gcn.preprocess_adj(A)
                except Exception:
                    A = self.model.gcn1.preprocess_adj(A)
                embedding = self.model.emb1(V)
                g1 = self.model.gcn1(embedding, A, preprocess_A=False)
                g2 = self.model.gcn2(g1, A, preprocess_A=False)
                new_v = torch.cat([g1, g2], dim=-1)
                new_v = self.model.emb2(new_v)
                batch_size, num_nodes, embedding_size = new_v.size()
                representation = new_v.view(batch_size * num_nodes, embedding_size)
                representations.append(representation.cpu())
                labels.append(targets.cpu())

        # Concatenate all the representations and labels.
        representations = torch.cat(representations, dim=0).numpy()
        labels = torch.cat(labels, dim=-1).numpy()

        # Apply t-SNE to reduce dimensionality to 2D
        tsne = TSNE(n_components=2, random_state=42)
        reduced_representation = tsne.fit_transform(representations)

        # Plot the 2D representation space
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced_representation[:, 0],
            reduced_representation[:, 1],
            c=labels,
            cmap="jet",
            alpha=0.6
        )
        plt.colorbar(scatter, label="Class Labels")
        plt.title("2D Visualization of Representation Space using t-SNE")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.savefig(f"{self.config.experiment_name}-representation_space.jpg")

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
