import collections
import os
import string
import warnings
from pathlib import Path

import pytest
from pytest import fixture

# Only test models which support training mode.
SUPPORTED_MODELS = ["model_1", "model_2", "model_3"]
FRAMEWORK_TO_MODELS_MAP = {
    "tf1": ["model_1", "model_3"],
    "torch": ["model_2"]}

# Number of top items from state_dict to consider 2 state_dict are the same.
SAME_N_ITEMs = 10
# TODO: Enable for different model backbone if needed.


def get_framework_from_model_name(model_name):
    for model_fw in FRAMEWORK_TO_MODELS_MAP.keys():
        if model_name in FRAMEWORK_TO_MODELS_MAP[model_fw]:
            return model_fw


def _clone_state_dict_from_model(model,
                                 model_framework: str = "torch",
                                 n_items: int = 10):
    """
    Clone first `check_layers` of state_dict.
    Args:
        model: Cassia Model. E.g: JeffLayout, TadashiLayout.
        model_framework: "torch" or "tf1"
        n_items: Number of early items to clone
    Returns:
        state_dict object with new memory allocation (clone) and difference graph history (detach)
    """

    assert n_items >= 0, f"`n_layers` must >= 0, but got {n_items}"

    if model_framework == "torch":
        i_item = 0
        new_state_dict = collections.OrderedDict()
        state_dict = model.model.state_dict()
        for key_item in state_dict.keys():
            if i_item >= n_items:
                break

            new_state_dict[key_item] = state_dict[key_item].detach().clone()
            i_item += 1

        return new_state_dict

    elif model_framework == "tf1":
        import tensorflow as tf

        state_dict = collections.OrderedDict()
        param_names = tf.trainable_variables()[:n_items]
        param_values = model.sess.run(param_names)

        for param_name, param_value in zip(param_names, param_values):
            state_dict[param_name] = param_value.copy()

        return state_dict


def _is_same_state_dict(state_dict_1, state_dict_2,
                        model_framework: str = "torch",
                        check_layers: int = 10):
    """
    Check if 2 state_dict is same content.
    Args:
        state_dict_1: state_dict of model_1/optimizer_1
        state_dict_2: state_dict of model_2/optimizer_2
        check_layers: Number of early layers to check
    Returns:
        True: if top check_layers of 2 state_dict is the same
        False: else
    """
    i_checked = 0
    for key_item_1, key_item_2 in zip(state_dict_1.items(), state_dict_2.items()):
        i_checked += 1
        if check_layers and i_checked > check_layers:
            break
        if model_framework == "torch":
            import torch

            if not torch.equal(key_item_1[1], key_item_2[1]):
                return False
        elif model_framework == "tf1":
            if not (key_item_1[1] == key_item_2[1]).all():
                return False
    return True


@fixture
def supported_models():
    import package_name
    classes = []

    for class_name in package_name.__all__:
        # TODO: Update "THEME_NAME"
        model_name: str = class_name.replace("THEME_NAME", "").lower()
        if model_name not in SUPPORTED_MODELS:
            warnings.warn(
                f'Training for {class_name} is not supported, skipping test training for {class_name}')
            continue
        framework = get_framework_from_model_name(model_name)

        classes.append(
            (getattr(package_name, class_name), framework)
        )

    return list(classes)


@fixture
def dataset():
    paths = Path(__file__).parent.glob('assets/inputs/*')
    # TODO: Update dataset format.
    labels = [(x, 'Dummy Label y') for x in paths]
    return zip(*labels)


def test_state_after_fit_function(tmpdir, supported_models, dataset):
    """
    Test if fit function change the internal state of model.
    """
    x, y = dataset

    for Model, model_fw in supported_models:
        model = Model(weights_path="", mode="training",
                      charset_list=string.printable)
        state_dict_before = _clone_state_dict_from_model(model.model.state_dict(),
                                                         model_framework=model_fw,
                                                         n_items=SAME_N_ITEMs)
        model.fit(x, y)
        state_dict_after = _clone_state_dict_from_model(model.model.state_dict(),
                                                        model_framework=model_fw,
                                                        n_items=SAME_N_ITEMs)
        assert _is_same_state_dict(state_dict_before, state_dict_after, check_layers=SAME_N_ITEMs) is False, \
            f"Model {Model.__name__} did not change state after calling fit()."


def test_state_and_output_after_validate_function(supported_models, dataset):
    """
    Test if validate function change the internal state of model.
    """
    x, y = dataset

    for Model, model_fw in supported_models:
        class_name = Model.__name__
        model = Model(weights_path=f"./models/{class_name}.pt")
        state_dict_before = _clone_state_dict_from_model(model.model.state_dict(),
                                                         model_framework=model_fw,
                                                         n_items=SAME_N_ITEMs)

        model.validate(x, y)
        state_dict_after = _clone_state_dict_from_model(model.model.state_dict(),
                                                        model_framework=model_fw,
                                                        n_items=SAME_N_ITEMs)
        assert _is_same_state_dict(state_dict_before, state_dict_after, check_layers=SAME_N_ITEMs) is True, \
            f"Model {Model.__name__} should not changed state after calling validate()."


def test_state_after_load_function(supported_models, dataset):
    """
    Test: init scratch -> load from previous weight, state dict should change.
    """

    for Model, model_fw in supported_models:
        model = Model(weights_path="", mode="training",
                      charset_list=string.printable)
        state_dict_before = _clone_state_dict_from_model(model.model.state_dict(),
                                                         model_framework=model_fw,
                                                         n_items=SAME_N_ITEMs)

        # Load pre-trained weights from dvc downloaded checkpoint
        model.load(f"./models/{Model.__name__}.pt")
        state_dict_after = _clone_state_dict_from_model(model.model.state_dict(),
                                                        model_framework=model_fw,
                                                        n_items=SAME_N_ITEMs)
        assert _is_same_state_dict(state_dict_before, state_dict_after, check_layers=SAME_N_ITEMs) is False, \
            f"Model {Model.__name__} did not load pre-trained weight from checkpoint."


def test_finetuning_with_fit_function(supported_models, dataset):
    """
    Test continue training from previous checkpoint.
    """
    x, y = dataset

    for Model, model_fw in supported_models:
        # Load pre-trained weights from dvc downloaded checkpoint
        model = Model(weights_path=f"./models/{Model.__name__}.pt",
                      mode="training", charset_list=string.printable)
        state_dict_before = _clone_state_dict_from_model(model.model.state_dict(),
                                                         model_framework=model_fw,
                                                         n_items=SAME_N_ITEMs)

        model.fit(x, y)
        state_dict_after = _clone_state_dict_from_model(model.model.state_dict(),
                                                        model_framework=model_fw,
                                                        n_items=SAME_N_ITEMs)
        assert _is_same_state_dict(state_dict_before, state_dict_after, check_layers=SAME_N_ITEMs) is False, \
            f"Model {Model.__name__} did not update after fine-tuning."


def test_load_and_attributes_from_saved_checkpoints(tmpdir, supported_models, dataset):
    """
    Test load checkpoint which was saved with save().
    And check optimizer attributes on save().
    """

    # TODO: Determine required_attributes.
    x, y = dataset
    required_attributes = ['model_state_dict',
                           'optimizer_state_dict', 'config', 'epoch']

    for Model, model_fw in supported_models:
        checkpoint_path = os.path.join(tmpdir, f"{Model.__name__}.pth")

        model = Model(weights_path=f"./models/{Model.__name__}.pt",
                      mode="training", charset_list=string.printable)
        model.fit(x, y)
        state_dict_before = _clone_state_dict_from_model(model.model.state_dict(),
                                                         model_framework=model_fw,
                                                         n_items=SAME_N_ITEMs)
        model.save(checkpoint_path)

        model = Model(weights_path=f"", mode="training",
                      charset_list=string.printable)
        model.load(checkpoint_path)
        # Predict for a single image
        model.process(x[0])

        state_dict_after = _clone_state_dict_from_model(model.model.state_dict(),
                                                        model_framework=model_fw,
                                                        n_items=SAME_N_ITEMs)
        assert _is_same_state_dict(
            state_dict_before, state_dict_after, check_layers=SAME_N_ITEMs) is True

        if model_fw == "torch":
            import torch
            package = torch.load(checkpoint_path)
            for attr in required_attributes:
                assert attr in package, f"Checkpoint from save() should include" \
                                        f" \"{attr}\", but only got={package.keys()}"


def test_load_and_attributes_from_exported_checkpoints(tmpdir, supported_models, dataset):
    """
    Test load checkpoint which was saved with export().
    """

    # TODO: Determine required_attributes.
    x, y = dataset
    required_attributes = ['model_state_dict', 'config']

    for Model, model_fw in supported_models:
        class_name = Model.__name__
        model = Model(weights_path=f"./models/{class_name}.pt",
                      mode="training", charset_list=string.printable)
        model.fit(x, y)

        state_dict_before = _clone_state_dict_from_model(model.model.state_dict(),
                                                         model_framework=model_fw,
                                                         n_items=SAME_N_ITEMs)
        checkpoint_path = os.path.join(tmpdir, f"{class_name}.pt")
        model.export(checkpoint_path)

        model = Model(weights_path=f"", mode="training",
                      charset_list=string.printable)
        model.load(checkpoint_path)
        # Predict for a single image
        model.process(x[0])

        state_dict_after = _clone_state_dict_from_model(model.model.state_dict(),
                                                        model_framework=model_fw,
                                                        n_items=SAME_N_ITEMs)
        assert _is_same_state_dict(
            state_dict_before, state_dict_after, check_layers=SAME_N_ITEMs) is True

        if model_fw == "torch":
            import torch
            package = torch.load(checkpoint_path)
            for attr in required_attributes:
                assert attr in package, f"Checkpoint from export() should include \"{attr}\", but only got={package.keys()}"


@pytest.mark.parametrize(['n_epochs', 'expected_current_epoch'], [(1, 1), (3, 3)])
def test_fit_and_resume_n_epochs(tmpdir, supported_models, dataset, n_epochs, expected_current_epoch):
    """
    Test running fit for multiple epochs and resume training.
    """
    x, y = dataset
    for Model, model_fw in supported_models:

        model = Model(weights_path="", mode="training",
                      charset_list=string.printable)
        for epoch in range(n_epochs):
            model.fit(x, y)

        config_before = model.config
        state_dict_after = _clone_state_dict_from_model(model.model.state_dict(),
                                                        model_framework=model_fw,
                                                        n_items=SAME_N_ITEMs)

        checkpoint_path = os.path.join(tmpdir, f"{Model.__name__}.pt")
        model.export(checkpoint_path)

        if model_fw == "torch":
            import torch
            package = torch.load(checkpoint_path)
            assert package["current_epoch"] == n_epochs

        # Load previous checkpoint and continue
        model_2 = Model(weights_path="", mode="training",
                        charset_list=string.printable)
        model_2.load(checkpoint_path)

        state_dict_load = _clone_state_dict_from_model(model_2.model.state_dict(),
                                                       model_framework=model_fw,
                                                       n_items=SAME_N_ITEMs)

        for epoch in range(n_epochs, expected_current_epoch):
            model_2.fit(x, y)

        state_dict_retrain = _clone_state_dict_from_model(model_2.model.state_dict(),
                                                          model_framework=model_fw,
                                                          n_items=SAME_N_ITEMs)

        assert _is_same_state_dict(state_dict_after, state_dict_load, model_framework=model_fw), \
            "Model state changed after clear and reload."
        assert not _is_same_state_dict(state_dict_load, state_dict_retrain, model_framework=model_fw), \
            "Reloaded model state does not change after fin-tuning with fit()."
        assert (model_2.current_epoch == expected_current_epoch), "model.current_epoch doesn't match with expected current_epoch. " \
            f"Current {model_2.current_epoch} != {expected_current_epoch} after fitted, saved, loaded, and fit."

        if model_fw == "torch":
            # Check config matching
            assert (config_before == model_2.config), "config should not change after load from saved checkpoints"

            # Check if optimizer ready for training (optimizer)
            assert (model_2.optimizer is not None), "model.optimizer is not initialized after load from saved checkpoints"
