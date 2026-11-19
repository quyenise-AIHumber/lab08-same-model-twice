"""Tests for the equivalent Keras network."""

import keras
import numpy as np
import pytest
import torch
from torch import nn

from keras_model import (
    build_keras_model,
    copy_torch_weights,
    get_keras_arrays,
    max_prediction_difference,
)
from model import BikeNet


# Test 1: Keras returns one prediction per row
def test_keras_prediction_shape():
    model = build_keras_model(n_features=8, hidden=32)

    x = np.ones((5, 8), dtype=np.float32)
    predictions = model(x, training=False)

    assert tuple(predictions.shape) == (5, 1)


# Test 2: parameter counts match for different sizes
@pytest.mark.parametrize("hidden", [4, 16, 32, 128])
def test_parameter_count_matches_pytorch(hidden):
    torch_model = BikeNet(n_features=8, hidden=hidden)
    keras_model = build_keras_model(n_features=8, hidden=hidden)

    torch_count = sum(p.numel() for p in torch_model.parameters())
    keras_count = keras_model.count_params()

    assert torch_count == keras_count


# Test 3: transferred kernels are correctly transposed
def test_weight_copy_transposes_kernels():
    torch.manual_seed(42)

    torch_model = BikeNet(8, hidden=32)
    keras_model = build_keras_model(8, hidden=32)

    torch_layers = [
        layer for layer in torch_model.modules() if isinstance(layer, nn.Linear)
    ]

    # Make the square middle matrix deliberately asymmetric.
    with torch.no_grad():
        torch_layers[1].weight.copy_(
            torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32) / 1024
        )

    copy_torch_weights(torch_model, keras_model)

    keras_layers = [
        layer for layer in keras_model.layers if isinstance(layer, keras.layers.Dense)
    ]

    for torch_layer, keras_layer in zip(torch_layers, keras_layers, strict=True):
        actual_kernel, actual_bias = keras_layer.get_weights()

        expected_kernel = torch_layer.weight.detach().numpy().T
        expected_bias = torch_layer.bias.detach().numpy()

        np.testing.assert_array_equal(actual_kernel, expected_kernel)

        np.testing.assert_array_equal(actual_bias, expected_bias)


# Test 4: Keras and PyTorch use identical prepared rows
def test_keras_uses_same_prepared_rows(prepared_data):
    for split in ("train", "val", "test"):
        x_keras, y_keras = get_keras_arrays(prepared_data, split)

        np.testing.assert_array_equal(x_keras, prepared_data[split].x.numpy())

        np.testing.assert_array_equal(y_keras, prepared_data[split].y.numpy())


# Test 5: predictions are equivalent on validation data
def test_predictions_match_after_weight_copy(
    prepared_data,
):
    torch.manual_seed(42)

    torch_model = BikeNet(8, hidden=32)
    keras_model = build_keras_model(8, hidden=32)

    copy_torch_weights(torch_model, keras_model)

    x_val, _ = get_keras_arrays(prepared_data, "val")

    difference = max_prediction_difference(
        torch_model,
        keras_model,
        x_val,
    )

    assert difference < 1e-5
