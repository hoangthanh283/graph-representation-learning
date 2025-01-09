import collections
from typing import Any, Dict, Tuple

import munch
import neptune.new as neptune
import torch
import torch.nn as nn

from gnn.trainer.training_procedures.kv_procedure import KVProcedure


class GraphClassificationProcedure(KVProcedure):
    def __init__(
        self,
        model: nn.Module,
        config: munch.munchify,
        neptune_exp: neptune.run.Run = None,
        **kwargs: Dict[str, Any],
    ):
        """Optmizing process for KV task."""
        super(GraphClassificationProcedure, self).__init__(
            model, config, neptune_exp, **kwargs
        )
        self.model = self._load_backbone(model)

    def _load_backbone(self, model: nn.Module) -> nn.Module:
        pretrained_path = self.config.optimize_settings.ssl_pretrain_path
        if pretrained_path:
            self.logger.info("Restoring pretrained checkpoint ...")
            checkpoint_dict = self.checkpointer.restore_checkpoint(pretrained_path)
            source_state = checkpoint_dict.get("state_dict", None)

            target_state = model.state_dict()
            new_target_state = collections.OrderedDict()

            for target_key, _ in target_state.items():
                if (
                    target_key in source_state
                    and source_state[target_key].size()
                    == target_state[target_key].size()
                ):
                    new_target_state[target_key] = source_state[target_key]
                else:
                    new_target_state[target_key] = target_state[target_key]
                    self.logger.warning(
                        "[WARNING] Not found pre-trained parameters for {}".format(
                            target_key
                        )
                    )

            try:
                model.load_state_dict(new_target_state)  # strict=False
            except RuntimeError:
                self.logger.debug(
                    "Could not load_state_dict by the normal way, \
                    retrying with DataParallel loading mode..."
                )
                model = torch.nn.DataParallel(model)
                model.load_state_dict(new_target_state)  # strict=False
                self.logger.debug("Loading Success!")

        else:
            self.logger.info("Not found any pretrained model!")

        return model

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
        batch_inputs = [
            batch_sample["textline_encoding"].float().to(self.device),
            batch_sample["adjacency_matrix"].float().to(self.device),
        ]

        # Set list of batch targets.
        batch_targets = batch_sample["graph_label"].to(self.device)

        # Run model prediction -- batch x nodes x classes.
        batch_predicts = self.model.forward(batch_inputs)

        # Calculate the loss.
        loss = self.criterion(batch_predicts, batch_targets.view(-1, 1))

        # Get the predicted class indexs,
        # batch x classes x nodes --> batch x nodes x classes.
        predicts = self.activator(batch_predicts)
        predicts = predicts.argmax(dim=-1)

        # Get metric scores.
        score_items, input_items = self._get_metric_scores(
            predicts, batch_targets, item_name="Node classification"
        )

        # Add metric scores.
        score_items.update({"loss": loss.item()})
        return loss, score_items, input_items
