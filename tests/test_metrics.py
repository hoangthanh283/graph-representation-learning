from pathlib import Path

from package_name.metrics import theme_metric
from pytest import fixture


@fixture
def dataset():
    json_path = str(Path(__file__).parent / 'assets' / 'samples')
    ca_json = f"{json_path}/ca.json"
    predicted = f"{json_path}/predicted.json"
    return ca_json, predicted


def test_theme_metrics(dataset):
    ca, predicted_out = dataset

    # TODO: Update the expected metric value.
    expected_output = None

    metric_value = theme_metric(predicted_out, ca)
    # assert metric_value == expected_output

    raise NotImplementedError
