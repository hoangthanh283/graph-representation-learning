from pathlib import Path
from typing import List

from package_name.evaluation import evaluate
from pytest import fixture

ROOT_FOLDER = Path(__file__).parent


@fixture
def dataset():
    # TODO: Define test file for this lib-<theme>
    json_path = str(Path(__file__).parent / 'samples')
    ca_json = f"{json_path}/ca.json"
    predicted = f"{json_path}/predicted.json"
    evaluated = f"{json_path}/evaluated.json"

    return ca_json, predicted, evaluated


def _same_eval_result(predicted_eval: List[dict], true_eval: List[dict]):
    for pred, true in zip(predicted_eval, true_eval):
        if pred != true:
            return False
    return True


def test_gt_str_pred_str(dataset):
    """
    Test evaluation, case:  predictions is path (str), targets is path (str)
    """
    target_path, predict_path, true_eval = dataset
    predicted_eval = evaluate(predictions=predict_path,
                              targets=target_path)
    assert _same_eval_result(predicted_eval, true_eval), \
        "Failed as predictions is path (str), targets is path (str)"

    assert isinstance(predicted_eval, dict),\
        f"Output of evaluate() should be dict, but got type={type(predicted_eval)}."
