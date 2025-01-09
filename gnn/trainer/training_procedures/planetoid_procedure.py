from collections import defaultdict
from typing import Any, Dict, Tuple

import munch
import neptune.new as neptune
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from tqdm import tqdm

from gnn.data_generator.base_dataloader import BaseDataLoader
from gnn.data_generator.datasets import PlanetoidDataset
from gnn.trainer.training_procedures.base_procedure import BaseProcedure
from gnn.utils.metric_tracker import Dictlist


class PlanetoidProcedure(BaseProcedure):
    GLOBAL_TEST_F1_SCORE = 0.0
    BEST_LAMBDA_VALUE = 0.0

    def __init__(self, model: nn.Module, config: munch.munchify,
                 neptune_exp: neptune = None, **kwargs: Dict[str, Any]):
        """Optmizing process for KV task. """
        super(PlanetoidProcedure, self).__init__(model, config, neptune_exp, **kwargs)
        self.global_step = 0
        self.config = config
        self.neptune_exp = neptune_exp
        self.activator = torch.nn.Softmax(dim=2)

        # Initialize dataloader/dataset base.
        dataloader = BaseDataLoader(self.config)
        self.dataset = PlanetoidDataset(self.config.data_config)
        self.dataloader = dataloader._get_dataloader(self.dataset, self.config.data_config)

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
        test_elements = [self.config.data_config.node_label_padding_value, self.config.data_config.other_class_index]
        selected_mask = np.isin(np_y_true, test_elements, invert=True)
        np_y_pred = np_y_pred[selected_mask]
        np_y_true = np_y_true[selected_mask]

        # Map index class to class names.
        pred_mapped_classes = [self.dataset.classes[idx] for idx in np_y_pred.tolist()]
        lbl_mapped_classes = [self.dataset.classes[idx] for idx in np_y_true.tolist()]

        # Get classification reports.
        try:
            score_matrix = classification_report(lbl_mapped_classes, pred_mapped_classes, output_dict=True,
                                                 zero_division=0)
            # Set score items with the item name.
            score_items = score_matrix["macro avg"]
        except Exception:
            score_items = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0.0}

        score_items = {"{0}_{1}".format(item_name, kk): vv for (kk, vv) in score_items.items()}
        # Return predicted values and ground-truth values.
        input_items = {"pred": pred_mapped_classes, "lbl": lbl_mapped_classes}
        return score_items, input_items

    def _step_process(self, batch_sample: torch.Tensor, mask: torch.Tensor,
                      **kwargs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """Step process for each batch sample.
        Args:
            batch_sample: Input batch samples.
            mask: To mask out nodes in the graph that is not required to be used.
        Return:
            loss: Loss value.
            metric_scores: Metric scores (f1 score, precision, recall, ... etc).
        """
        # Set list of batch inputs.
        batch_inputs = [
            batch_sample["nodes"].float().to(self.device).squeeze(),  # nodes x channels.
            batch_sample["edge_index"].to(self.device).squeeze(),  # 2 x num edges.
        ]

        # Remove batch dim and mask out non-training samples.
        targets = batch_sample["labels"].to(self.device).squeeze()
        masked_targets = targets[mask].unsqueeze(0)

        # Run model prediction --> batch x nodes x classes.
        outputs = self.model.forward(batch_inputs)
        masked_outputs = outputs[mask].unsqueeze(0)
        loss = self.criterion(masked_outputs, masked_targets)

        # logsoftmax_outs = F.log_softmax(outputs, dim=1)
        # label = logsoftmax_outs.max(1)[1]
        # label[mask] = masked_targets
        # label.requires_grad = False
        # loss = F.nll_loss(logsoftmax_outs[mask], label[mask])
        # # gamma = 1.0
        # # lambda_ = (kwargs["epoch"]/self.config.num_epochs)**gamma
        # # loss += lambda_ * F.nll_loss(logsoftmax_outs[~mask], label[~mask])

        # Get the predicted class indexs,
        # batch x classes x nodes --> batch x nodes x classes.
        predicts = self.activator(masked_outputs)
        predicts = predicts.argmax(dim=-1)

        # Get metric scores.
        score_items, input_items = self._get_metric_scores(predicts, masked_targets, item_name="Node classification")

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
            loss, metric_scores, input_items = self._step_process(batch_sample, self.dataset.graph_dataset.train_mask,
                                                                  **kwargs)
            loss.backward()

            # Clip the big gradient value, in the range of 0.0 to 1.
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        return metric_scores, input_items

    def _run_val_step(self, batch_sample: torch.Tensor, mask: torch.Tensor,
                      **kwargs: Dict[str, Any]) -> Tuple[Dict[int, Any], Dict[str, Any]]:
        """Run the batch validation step and calculate metric scores.

        Args:
            batch_sample: Dict of input samples.
            mask: To mask out nodes in the graph that is not required to be used.
        """
        # Run predicting.
        self.model.eval()
        with torch.set_grad_enabled(False):
            _, metric_scores, input_items = self._step_process(batch_sample, mask, **kwargs)
        return metric_scores, input_items

    def _optimize_per_epoch(self, epoch: int) -> Dict[int, Any]:
        """Run training step.

        Args:
            epoch: Ith epoch.

        Returns:
            val_metrics: Validation metric scores.
        """
        self._run_training_epoch(epoch)
        _, val_item_dict = self._run_eval_epoch(epoch, self.dataset.graph_dataset.val_mask, "Validation")
        self._log_classification_report(val_item_dict, epoch, "Validation")

        test_metrics, test_item_dict = self._run_eval_epoch(epoch, self.dataset.graph_dataset.test_mask, "Testing")
        self._log_classification_report(test_item_dict, epoch, "Testing")

        macro_avg_test = self._get_macro_avg_test(test_metrics, test_item_dict)
        return macro_avg_test

    def _run_training_epoch(self, epoch: int) -> Dict:
        train_metrics = Dictlist()
        training_bar = tqdm(self.dataloader, desc=f"Training - Epochs {epoch}")
        for train_batch in training_bar:
            tr_metric_scores, _ = self._run_train_step(train_batch, **{"epoch": epoch})
            train_metrics._update(tr_metric_scores)
            self.tb_writer.add_scalar("Train_step_loss", tr_metric_scores["loss"], self.global_step)
            if self.neptune_exp:
                self.neptune_exp["Train/step_loss"].append(tr_metric_scores["loss"])
            training_bar.set_description(f"Epoch {epoch} - Train/Step loss: {tr_metric_scores['loss']}")
            self.model.zero_grad()
            self.global_step += 1
            # self.model.lambda_value = self.cosine_schedule_lambda(
            #     self.global_step, epoch, self.config.num_epochs * len(self.dataloader),
            #     base_value=0.01, max_value=5.0, warmup_steps=5 * len(self.dataloader)
            # )

        train_metrics = train_metrics._result()
        self.logger.info(f"Training epoch: {epoch} step: {self.global_step} metrics: {train_metrics}")
        for metric_name, score in train_metrics.items():
            self.tb_writer.add_scalar(f"Train {metric_name}", score, epoch)
            if self.neptune_exp:
                self.neptune_exp[f"Train/{metric_name}"].append(score)
        return train_metrics

    def _run_eval_epoch(self, epoch: int, mask: torch.Tensor, prefix_name: str = "Test"
                        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        eval_metrics = Dictlist()
        eval_item_dict = defaultdict(list)
        eval_bar = tqdm(self.dataloader, desc=f"{prefix_name} - Epochs {epoch}")
        for eval_batch in eval_bar:
            eval_metric_scores, val_input_items = self._run_val_step(eval_batch, mask, **{"epoch": epoch})
            eval_metrics._update(eval_metric_scores)
            eval_bar.set_description(f"Epoch {epoch} - {prefix_name}/Step loss: {eval_metric_scores['loss']}")
            for kn, item_values in val_input_items.items():
                eval_item_dict[kn].extend(item_values)

        eval_metrics = eval_metrics._result()
        self.logger.info(f"{prefix_name} metrics: {eval_metrics}")
        for metric_name, score in eval_metrics.items():
            self.tb_writer.add_scalar(f"Val {metric_name}", score, epoch)
            if self.neptune_exp:
                self.neptune_exp[f"{prefix_name}/{metric_name}"].append(score)
        return eval_metrics, eval_item_dict

    def _log_classification_report(self, item_dict: Dict, epoch: int, phase: str):
        report = classification_report(item_dict["lbl"], item_dict["pred"])
        report_dict = classification_report(item_dict["lbl"], item_dict["pred"], output_dict=True)
        macro_avg = report_dict["macro avg"]
        for macro_metric_name, macro_score in macro_avg.items():
            self.tb_writer.add_scalar(f"Macro {phase} {macro_metric_name}", macro_score, epoch)
            if self.neptune_exp:
                self.neptune_exp[f"Macro {phase}/{macro_metric_name}"].append(macro_score)
            if macro_metric_name == "f1-score" and macro_score > self.GLOBAL_TEST_F1_SCORE:
                self.GLOBAL_TEST_F1_SCORE = macro_score
                self.BEST_LAMBDA_VALUE = self.model.lambda_param

        self.logger.info(f"{phase} Classification report")
        self.logger.info(f"\n{report}")

    def _get_macro_avg_test(self, test_metrics: Dict, test_item_dict: Dict) -> Dict:
        test_report_dict = classification_report(test_item_dict["lbl"], test_item_dict["pred"], output_dict=True)
        macro_avg_test = test_report_dict["macro avg"]
        macro_avg_test["loss"] = test_metrics["loss"]
        return macro_avg_test

    def __call__(self) -> str:
        """Optimizing process. """
        best_loss = float("inf")
        self.logger.info("Start optimizing ...")
        self.model.zero_grad()
        for epoch in range(self.config.num_epochs):
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

        self.logger.info(f"GLOBAL_TEST_F1_SCORE: {self.GLOBAL_TEST_F1_SCORE}")
        self.logger.info(f"BEST_LAMBDA_VALUE: {self.BEST_LAMBDA_VALUE}")
        return metrics["f1-score"]
