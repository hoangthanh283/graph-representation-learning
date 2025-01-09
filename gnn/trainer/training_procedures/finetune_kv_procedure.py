import collections
from typing import Any, Dict

import munch
import neptune.new as neptune
import torch
import torch.nn as nn

from gnn.trainer.training_procedures.kv_procedure import KVProcedure


class FinetuneKVProcedure(KVProcedure):
    def __init__(
        self,
        model: nn.Module,
        config: munch.munchify,
        neptune_exp: neptune.run.Run = None,
        **kwargs: Dict[str, Any],
    ):
        """Optmizing process for KV task."""
        super(FinetuneKVProcedure, self).__init__(model, config, neptune_exp, **kwargs)
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
