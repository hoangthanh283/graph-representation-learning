import argparse

from gnn.cl_warper import GNNLearningWarper
from gnn.models import RPGCN, DeepRPGCN, DeepRPRobustGCN, RPGraphCNNDropEdge

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CL configurations")
    parser.add_argument("--config", default=None, type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    # Model details.
    # EXP_POST_FIX = "DeepRPRobustGCN-rp-layers-1-4-7-final"
    # EXP_POST_FIX = "DeepRPRobustGCN-rp-layer-at-last-scheduling"
    # EXP_POST_FIX = "DeepRPRobustGCN-rp-layer-at-last-cosine-scheduling"
    # EXP_POST_FIX = "DeepRPGCN-baseline-w/o-skip-connection-dropout-attn-32-layers"
    # EXP_POST_FIX = "DeepRPGCN-baseline-w/o-skip-connection-dropout-attn-32-layers-rp-layer-after-3-gcn-layers"
    # EXP_POST_FIX = "DeepRPGCN-baseline-w/o-skip-connection-dropout-attn-32-layers-rp-layer-before-last-layer"
    # EXP_POST_FIX = "DeepRPGCN-baseline-w/o-skip-connection-dropout-attn-32-layers-rp-layer-after-5-gcn-layers"
    # EXP_POST_FIX = "DeepRPGCN-baseline-w/o-skip-connection-dropout-attn-32-layers-rp-layer-after-7-gcn-layers"
    # EXP_POST_FIX = "DeepRPGCN-baseline-w/o-skip-connection-dropout-attn-32-layers-rp-layer-after-10-gcn-layers"
    # EXP_POST_FIX = "DeepRPGCN-baseline-32-gcn-layers"
    # EXP_POST_FIX = "DeepRPGCN-baseline-w/o-skip-connection-dropout-attn-rp-layers-1-4-7-final"

    EXP_POST_FIX = "DeepRPRobustGCN-rp-layer-at-last-cosine-scheduling"

    rp_size = None
    lambda_value = 1  # 0.01  # 0.005
    # model = DeepRPRobustGCN(input_dim=4369, output_dim=45, num_edges=6, net_size=256, rp_size=rp_size,
    #                         lambda_value=lambda_value)
    # model = DeepRPGCN(input_dim=4369, output_dim=45, num_edges=6, net_size=256, rp_size=rp_size,
    #                   lambda_value=lambda_value)
    model = RPGCN(input_dim=1433, output_dim=7, net_size=256, rp_size=rp_size, lambda_value=lambda_value, use_attention=False)

    # Write experiment names.
    config = GNNLearningWarper._from_config(args.config)
    config.experiment_name = f"{config.experiment_name}-{config.data_config.dataset_name}-rp_size-{rp_size}-lambda-{lambda_value}-{EXP_POST_FIX}"

    # Define the optimzation warper.
    warper = GNNLearningWarper(model, config=config)
    warper.train()
