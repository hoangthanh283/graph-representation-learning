from typing import Any, Dict, Optional, Tuple
from tabulate import tabulate

import torch
from neptune import Run as NeptuneRun
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm

from gnn.data_generator.datasets.planetoid_dataset import PlanetoidDatasetName, get_planetoid_dataset
from gnn.models.networks.deep_rp_planetoid_gcn import DeepRPPlanetoidGCN
from gnn.trainer.training_procedures.base_procedure import BaseProcedure
from gnn.utils.metric_tracker import Dictlist


class PlanetoidProcedure(BaseProcedure):
    def __init__(self, model: DeepRPPlanetoidGCN, config: Any,
                 ems_exp: Optional[NeptuneRun] = None, **kwargs: Dict[str, Any]):
        """Optmizing process for KV task. """
        super(PlanetoidProcedure, self).__init__(model, config, ems_exp, **kwargs)
        self.global_step = 0
        self.config = config
        self.ems_exp = ems_exp
        self.planetoid_dataset = self._init_dataloaders()

    def _init_dataloaders(self):
        """
        Load planetoid dataset.
        """
        # TODO: We should define the config type more clearly.
        dataset_name = PlanetoidDatasetName[self.config.data_config.type]
        dataset = get_planetoid_dataset(dataset_name)
        return dataset

    def _train_step(self) -> Dict[str, float]:
        """
        Perform a training step.
        :return: Training step report.
        """
        self.model.train()
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            predicts = self.model(inputs=(self.planetoid_dataset.x, self.planetoid_dataset.adj_matrix))
            loss, train_f1 = self.step_report(predicts, self.planetoid_dataset.train_mask)
            loss.backward()

            # Clip the big gradient value, in the range of 0.0 to 1.
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        with torch.no_grad():
            _, dev_f1 = self.step_report(predicts, self.planetoid_dataset.val_mask)
            _, test_f1 = self.step_report(predicts, self.planetoid_dataset.test_mask)
        return {"loss": float(loss.data), "train_f1": train_f1, "dev_f1": dev_f1, "test_f1": test_f1}

    def step_report(self, predicts: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Calculate evaluation report.
        """
        labels = self.planetoid_dataset.y
        masked_predicts = predicts[:, mask]
        masked_labels = labels[:, mask]
        loss = self.criterion(masked_predicts, masked_labels)
        decoded_predicts = masked_predicts.argmax(dim=-1)
        return loss, f1_score(masked_labels.cpu()[0], decoded_predicts.cpu()[0], average="weighted")

    def run_train(self, num_epoch: int) -> Dict[int, Any]:
        """
        Train model.
        """
        train_metrics = Dictlist()
        for epoch in tqdm(range(num_epoch)):
            tr_metric_scores = self._train_step()
            train_metrics._update(tr_metric_scores)
            self.tb_writer.add_scalar("Train_step_loss", tr_metric_scores["loss"], self.global_step)
            if self.ems_exp:
                self.ems_exp["Train/step_loss"].append(tr_metric_scores["loss"])
            self.global_step += 1
            self.model.lambda_value = self.cosine_schedule_lambda(self.global_step, epoch,
                                                                  self.config.num_epochs * num_epoch,
                                                                  base_value=1e-4, max_value=1.0,
                                                                  warmup_steps=5 * num_epoch)
            table_data = [["Epoch", epoch]] + [[key, value] for key, value in tr_metric_scores.items()]
            table = tabulate(table_data,
                             headers=[],
                             tablefmt='simple')
            print(table)
