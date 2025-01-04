import argparse

from gnn.cl_warper import GNNLearningWarper
from gnn.data_generator.datasets.planetoid_dataset import PlanetoidDatasetName, get_planetoid_dataset
from gnn.trainer.training_procedures.planetoid_procedure import PlanetoidProcedure
from gnn.models.networks.deep_rp_robust_gcn import DeepRPRobustGCN

parser = argparse.ArgumentParser(description="CL configurations")
parser.add_argument("--config", default=None, type=str, help="Path to the configuration file.")
args = parser.parse_args()

EXP_POST_FIX = "DeepRPRobustGCN-rp-layer-at-last-cosine-scheduling"

dataset = get_planetoid_dataset(PlanetoidDatasetName.CORA)

cora_dataset = dataset

rp_size = None
lambda_value = 1  # 0.01  # 0.005
model = DeepRPRobustGCN(input_dim=cora_dataset.get_input_dim(), output_dim=cora_dataset.get_num_classes(), num_edges=1,
                        lambda_value=lambda_value)

# Write experiment names.
config = GNNLearningWarper._from_config(args.config)
procedure = PlanetoidProcedure(model, config)

procedure.run_train(num_epoch=200)
