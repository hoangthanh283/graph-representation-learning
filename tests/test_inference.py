from pytest import fixture


@fixture
def installed_models():
    return None


@fixture
def dataset():
    return None


def test_single_inference(installed_models, dataset):
    raise NotImplementedError


def test_batch_inference(installed_models, dataset):
    raise NotImplementedError
