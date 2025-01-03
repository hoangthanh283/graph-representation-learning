import argparse

import munch
from bayes_opt import BayesianOptimization

from gnn.cl_warper import GNNLearningWarper
from gnn.models import RPRobustFilterGraphCNNDropEdge


def objective_function(lambda_value: float, rp_size: int, config: munch.munchify) -> float:
    """
    Objective function for Bayesian optimization.
    :param lambda_value: float, lambda value for regularization.
    :param rp_size: int, random projection size.
    :param config: munch.munchify, configuration parameters.
    :return: float, f1 score.
    """
    # Create model with current parameters
    model = RPRobustFilterGraphCNNDropEdge(
        input_dim=4369,
        output_dim=45,
        num_edges=6,
        net_size=256,
        rp_size=int(rp_size),
        lambda_value=lambda_value
    )
    # Define the optimzation warper.
    warper = GNNLearningWarper(model, config=config)
    f1_score = warper.train()
    return f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CL configurations")
    parser.add_argument("--config", default=None, type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    # Model details.
    EXP_POST_FIX = "rp-layer-after-embed2-before-last-layer-with-bayes-optimization"
    # EXP_POST_FIX = "rp-layer-after-embed2-layer-with-bayes-optimization"

    config = GNNLearningWarper._from_config(args.config)
    config.experiment_name = f"{config.experiment_name}-{EXP_POST_FIX}"

    # Setup bounds for optimization
    bounds = {
        "lambda_value": (0, 1),
    }
    optimizer = BayesianOptimization(
        f=lambda **params: objective_function(params["lambda_value"], 128, config),
        pbounds=bounds,
        random_state=1234
    )

    # Optimize
    optimizer.maximize(init_points=5, n_iter=15)

    # Get best parameters
    best_lambda = optimizer.max["params"]["lambda_value"]
    print(f"Best parameters: lambda={best_lambda}")
