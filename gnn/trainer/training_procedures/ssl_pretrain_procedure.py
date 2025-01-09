from typing import Any, Dict, List, Tuple

import munch
import neptune.new as neptune
import torch
import torch.nn as nn

from gnn.models.networks.dgi import DGI
from gnn.trainer import losses
from gnn.trainer.training_procedures.kv_procedure import KVProcedure


class SSLPretrainProcedure(KVProcedure):
    def __init__(
        self,
        model: nn.Module,
        config: munch.munchify,
        tasks: List,
        neptune_exp: neptune.run.Run = None,
        **kwargs: Dict[str, Any],
    ):
        """Optmizing process for KV task."""
        super(SSLPretrainProcedure, self).__init__(model, config, neptune_exp, **kwargs)
        self.tasks = tasks
        self.criterions = {
            "node_property": losses.MSELoss(),
            "edge_mask": losses.BinaryCrossEntropyLoss(),
            "pairwise_distance": losses.CrossEntropyLoss(),
            "pairwise_similarity": losses.MSELoss(),
            "graph_edit_distance": losses.MSELoss(),
            "dgi": losses.BinaryCrossEntropyLoss()
        }
        self.dgi_pretrainer = DGI(encoder=self.model, output_dim=self.config.network.args.net_size // 2).to(self.device)

    def _step_process(self, batch_sample: torch.Tensor, **kwargs: Dict[str, Any]
                      ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Step process for each batch sample.

        Args:
            batch_sample: Input batch samples.

        Return:
            loss: Loss value.
            metric_scores: Metric scores (f1 score, precision, recall, ... etc).
        """
        # Set list of batch inputs.

        total_loss = []
        for task in self.tasks:
            if task == "node_property":
                batch_inputs = [
                    batch_sample["textline_encoding"].float().to(self.device),
                    batch_sample["adjacency_matrix"].float().to(self.device),
                ]
                # Set list of batch targets.
                batch_targets = batch_sample[task].to(self.device)

                # Run model prediction -- batch x nodes x classes.
                batch_predicts = self.model.forward(batch_inputs, task=task)
                loss = self.criterions[task](batch_predicts, batch_targets)
                total_loss.append(loss)
            elif task in ["edge_mask", "pairwise_distance", "pairwise_similarity"]:
                batch_inputs = [
                    batch_sample["textline_encoding"].float().to(self.device),
                    batch_sample["adjacency_matrix"].float().to(self.device),
                ]
                batch_edges, batch_targets = (
                    batch_sample[task + "_indices"],
                    batch_sample[task + "_targets"],
                )
                batch_targets = batch_targets.to(self.device)

                batch_predicts = self.model.forward(
                    batch_inputs, batch_edges, task=task
                )
                loss = self.criterions[task](batch_predicts, batch_targets)
                total_loss.append(loss)
            elif task in ["graph_edit_distance"]:
                batch_inputs = [
                    batch_sample["textline_encoding"].float().to(self.device),
                    batch_sample["adjacency_matrix"].float().to(self.device),
                    batch_sample["aug_textline_encoding"].float().to(self.device),
                    batch_sample["aug_adjacency_matrix"].float().to(self.device),
                ]
                batch_targets = batch_sample[task].to(self.device)

                batch_predicts = self.model.forward(batch_inputs, task=task)
                loss = self.criterions[task](batch_predicts, batch_targets)
                total_loss.append(loss)
            elif task in ["dgi"]:
                batch_inputs = [
                    batch_sample["textline_encoding"].float().to(self.device),
                    batch_sample["adjacency_matrix"].float().to(self.device),
                    batch_sample["negative_textline_encoding"].float().to(self.device),
                    batch_sample["negative_adjacency_matrix"].float().to(self.device),
                ]
                batch_targets = batch_sample[task].to(self.device)
                batch_node_emb, batch_neg_node_emb = self.model.forward(batch_inputs, task=task)
                batch_predicts = self.dgi_pretrainer.forward_contrastive(batch_node_emb, batch_neg_node_emb)
                loss = self.criterions[task](batch_predicts, batch_targets)
                total_loss.append(loss)

        loss = sum(total_loss)
        # Calculate the loss.

        # Get the predicted class indexs,
        # batch x classes x nodes --> batch x nodes x classes.
        batch_inputs = [
            batch_sample["textline_encoding"].float().to(self.device),
            batch_sample["adjacency_matrix"].float().to(self.device),
        ]
        batch_targets = batch_sample["node_label"].to(self.device)
        batch_predicts = self.model.forward(batch_inputs)
        predicts = self.activator(batch_predicts)
        predicts = predicts.argmax(dim=-1)

        # Get metric scores.
        score_items, input_items = self._get_metric_scores(
            predicts, batch_targets.long(), item_name="Node classification"
        )

        # Add metric scores.
        score_items.update({"loss": loss.item()})
        return loss, score_items, input_items
