from pathlib import Path

from jsonschema import ValidationError, validate
from pytest import fixture


@fixture
def installed_models():
    # TODO: Init all available models.
    # return [Inited_models, ..., ...]
    raise NotImplementedError


@fixture
def dataset():
    import json

    def load_json(f_path):
        with open(f_path, "r", encoding="utf8") as file:
            return json.load(file)

    # TODO: Define test files for this lib-<theme>
    # For schema setting tutorial, visit: https://json-schema.org/learn/getting-started-step-by-step

    json_path = str(Path(__file__).parent / 'assets' / 'samples')

    data_in = load_json(f"{json_path}/input.json")
    predicted = load_json(f"{json_path}/predicted.json")
    # ca_json = load_json(f"{json_path}/ca.json")

    schema_path = str(Path(__file__).parent / 'assets' / 'schemas')

    schema_in = load_json(f"{schema_path}/input_schema.json")
    schema_out = load_json(f"{schema_path}/output_schema.json")
    # schema_label = load_json(f"{schema_path}/label_schema.json")

    # return data_in, predicted, ca_json, schema_in, schema_out, schema_label
    return data_in, predicted, schema_in, schema_out


def test_valid_example_data_json_schema(dataset):
    """
        Validate weather the example output.json / label_result.json file
         matches the standard json output schema for lib-<package_name>.
    """
    data_in, predicted, ca_json = dataset

    assert is_valid_json_schema(
        data_in, in_schema), f"Lib example input {data_in} didn't match lib input schema."
    assert is_valid_json_schema(
        predicted, out_schema), f"Lib example output {predicted} didn't match lib output schema."
    # assert is_valid_json_schema(ca_json, label_schema), f"Lib example label {ca_json} didn't match lib label schema."


def test_valid_single_output_schema(dataset, installed_models):
    """
        Validate weather the model-processed single output matches the standard json output schema for lib-<package_name>.
    """
    data_in, _, _ = dataset

    for Model_instance in installed_models:
        single_output = Model_instance.process(data_in)
        assert (is_valid_json_schema(single_output, out_schema)), \
            f"Model {Model_instance.__name__}'s single output did not match the lib output schema."


def test_valid_multiple_output_schema(dataset, installed_models):
    """
        Validate weather the model-processed multiple output matches the standard json output schema for lib-kv.
    """
    data_in, _, _ = dataset

    for Model_instance in installed_models:
        single_output = Model_instance.process([data_in, data_in, data_in])
        assert (is_valid_json_schema(single_output, out_schema)), \
            f"Model {Model_instance.__name__}'s multiple output did not match the lib output schema."


def is_valid_json_schema(x, schema):
    """
        Validate weather the "dict x" matches our expected "json schema".
    :param x: input to check schema.
     type(x):
      dict or list(dict)
    :param schema:
    The json schema to match.
     type(schema) = dict
    :return: bool
    """
    if isinstance(x, dict):
        x = [x]
    try:
        validate(x, schema)
    except ValidationError:
        return False
    return True
