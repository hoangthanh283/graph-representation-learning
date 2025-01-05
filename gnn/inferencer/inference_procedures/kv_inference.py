import gc
from typing import Any, Dict, List, Union

import munch
import torch
import torch.nn as nn
from tqdm import tqdm

from gnn.data_generator.base_dataloader import BaseDataLoader
from gnn.inferencer.inference_procedures import BaseProcedure
from gnn.utils.input_wrapper import cast_label_to_list, handle_single_input


class KVInference(BaseProcedure):
    def __init__(self, model: nn.Module, config: munch.munchify, **kwargs: Dict[str, Any]):
        """Key-value inference base procedure.

        Args:
            model: A network instance.
            config: Configuration parameters.
        """
        super(KVInference, self).__init__(model, config, **kwargs)
        self.infer_config = config.inference_settings
        act_config = self.infer_config.activation
        self.activator = getattr(torch.nn, act_config.type)(**act_config.args)
        self.dataloader = BaseDataLoader(self.config)
        self.dataset = self.dataloader._load_dataset(self.config.inference_settings.datasets.type,
                                                     self.config.inference_settings.datasets.args,
                                                     data_type="inference")
        self.id_to_class = self.dataset.id_to_class
        self.id_to_class[0] = ("other", "other")

    def step_process(self, batch_sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Step process for each batch sample.

        Args:
            batch_sample: Input batch samples.

        Return:
            Predicted output batch.
        """
        # Set list of batch inputs.
        raw_input = batch_sample["label"]
        batch_inputs = [
            torch.tensor(batch_sample["textline_encoding"], dtype=torch.float, device=self.device).unsqueeze(0),
            torch.tensor(batch_sample["adjacency_matrix"], dtype=torch.float, device=self.device).unsqueeze(0)
        ]

        # Run model prediction -- batch x nodes x classes.
        batch_predicts = self.model(batch_inputs)

        # Get the predicted class indexs,
        # batch x classes x nodes --> batch x nodes x classes.
        logits = self.activator(batch_predicts)
        logit_scores, class_indices = logits.max(dim=-1)

        # Flatten predictions into numpy array.
        flatten_class_indices = torch.reshape(class_indices, shape=(-1,))
        class_index_list = flatten_class_indices.cpu().detach().numpy().tolist()

        flatten_logit_scores = torch.reshape(logit_scores, shape=(-1,))
        logit_score_list = flatten_logit_scores.cpu().detach().numpy().tolist()

        # Map index class to class names.
        pred_mapped_classes = [self.id_to_class[idx] for idx in class_index_list]

        # Get entity's classes (Remove inner lists).
        assert len(raw_input) == len(pred_mapped_classes) == len(logit_score_list)
        outputs: List[Dict[str, Any]] = []
        for idx, (pair_labels, score) in enumerate(zip(pred_mapped_classes, logit_score_list)):
            box = raw_input[idx]
            formal_key, key_type = pair_labels
            box["key_type"] = key_type
            box["formal_key"] = formal_key
            box["confidence"] = score
            outputs.append(box)

        return outputs

    @handle_single_input(cast_label_to_list)
    def __call__(self, samples: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """Run prediction.

        Args:
            samples: Input samples. Each sample should have a format as below:
                [
                    {"location": [[0, 0], [0, 0], [0, 0], [0, 0]], "text": "abc"},
                    {"location": [[0, 0], [0, 0], [0, 0], [0, 0]], "text": "def"},
                    ...
                ]

        Return:
            Predicted entities. Each output sample would have a format as below:
                [
                    {"location": [[0, 0], [0, 0], [0, 0], [0, 0]], "text": "abc",
                     "key_type": "key", "confidence": 0.99, "formal_key": entity_name},

                    {"location": [[0, 0], [0, 0], [0, 0], [0, 0]], "text": "edf",
                     "key_type": "value", "confidence": 0.99, "formal_key": entity_name},
                    ...
                ]
        """
        torch.cuda.empty_cache()
        self.logger.info("Start processing ...")
        self.model.eval()

        # Initialize data-loader.
        # is_nested_list = any(isinstance(sub, list) for sub in samples)
        # samples = [samples] if not is_nested_list else samples
        self.dataset.list_samples = self.dataset._load_samples(samples)

        # Run predicting.
        outputs: List[Dict[str, Any]] = []
        for batch_sample in tqdm(self.dataset, total=len(self.dataset)):
            preds = self.step_process(batch_sample)
            outputs.append(preds)

        gc.collect()
        return outputs
