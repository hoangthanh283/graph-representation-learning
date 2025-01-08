import argparse

from gnn.cl_warper import GNNLearningWarper
from gnn.data_generator.datasets.planetoid_dataset import PlanetoidDatasetName, get_planetoid_dataset
from gnn.trainer.training_procedures.planetoid_procedure import PlanetoidProcedure
from gnn.models.networks.deep_rp_planetoid_gcn import DeepRPPlanetoidGCN

parser = argparse.ArgumentParser(description="CL configurations")
parser.add_argument("--config", default=None, type=str, help="Path to the configuration file.")
args = parser.parse_args()

EXP_POST_FIX = "Planetoid-simple"

dataset = get_planetoid_dataset(PlanetoidDatasetName.CORA)

cora_dataset = dataset

rp_size = None
lambda_value = 1  # 0.01  # 0.005
model = DeepRPPlanetoidGCN(input_dim=cora_dataset.get_input_dim(), output_dim=cora_dataset.num_classes,
                           lambda_value=lambda_value)

# Write experiment names.
config = GNNLearningWarper._from_config(args.config)
config.experiment_name = f"{config.experiment_name}-rp_size-{rp_size}-lambda-{lambda_value}-{EXP_POST_FIX}"

procedure = PlanetoidProcedure(model, config)

procedure.run_train(num_epoch=config.num_epochs)
