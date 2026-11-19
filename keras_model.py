"""Keras equivalent of the PyTorch BikeNet from Lab 07."""

import keras
import numpy as np
import torch
from torch import nn


# 1. Build the equivalent Keras network
def build_keras_model(n_features: int, hidden: int = 32):
    if n_features < 1 or hidden < 1:
        raise ValueError("Feature and hidden sizes must be positive.")

    return keras.Sequential(
        [
            keras.Input(shape=(n_features,), dtype="float32"),
            keras.layers.Dense(hidden, activation="relu", name="hidden_1"),
            keras.layers.Dense(hidden, activation="relu", name="hidden_2"),
            keras.layers.Dense(1, activation=None, name="rentals"),
        ],
        name="bike_demand_keras",
    )


# 2. Share the SAME prepared data with Keras
def get_keras_arrays(data, split):
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown data split.")

    dataset = data[split]

    x = dataset.x.detach().cpu().numpy().copy()
    y = dataset.y.detach().cpu().numpy().copy()

    return x, y


# 3. Copy weights from PyTorch into Keras
def copy_torch_weights(torch_model, keras_model):
    torch_layers = [
        layer for layer in torch_model.modules() if isinstance(layer, nn.Linear)
    ]

    keras_layers = [
        layer for layer in keras_model.layers if isinstance(layer, keras.layers.Dense)
    ]

    if len(torch_layers) != 3 or len(keras_layers) != 3:
        raise ValueError("Expected three dense layers per model.")

    for torch_layer, keras_layer in zip(torch_layers, keras_layers, strict=True):
        # PyTorch: (outputs, inputs)
        # Keras:   (inputs, outputs)
        kernel = torch_layer.weight.detach().cpu().numpy().T.copy()

        # Bias is a one-dimensional vector
        bias = torch_layer.bias.detach().cpu().numpy().copy()

        if tuple(keras_layer.kernel.shape) != kernel.shape:
            raise ValueError("Layer weight shapes do not match.")

        if tuple(keras_layer.bias.shape) != bias.shape:
            raise ValueError("Layer bias shapes do not match.")

        keras_layer.set_weights([kernel, bias])


# 4. Check prediction equivalence
def max_prediction_difference(
    torch_model,
    keras_model,
    features,
    atol=1e-5,
    rtol=1e-5,
):
    features = np.asarray(features, dtype=np.float32)

    # PyTorch inference
    torch_model.eval()

    with torch.no_grad():
        torch_predictions = torch_model(torch.from_numpy(features)).cpu().numpy()

    # Keras inference: shape (N, 1) -> (N,)
    keras_predictions = keras.ops.convert_to_numpy(
        keras_model(features, training=False)
    ).reshape(-1)

    # Fail clearly if the predictions differ
    np.testing.assert_allclose(
        torch_predictions,
        keras_predictions,
        atol=atol,
        rtol=rtol,
    )

    return float(np.max(np.abs(torch_predictions - keras_predictions)))


# 5. Standalone verification
if __name__ == "__main__":
    from model import BikeNet
    from prepare import prepare_data

    data = prepare_data(seed=42)

    x_val, y_val = get_keras_arrays(data, "val")
    n_features = data["train"].x.shape[1]

    print("Input features:", n_features)
    print("Validation rows:", len(y_val))

    for hidden in (32, 128):
        print(f"\nChecking hidden size {hidden}")

        torch.manual_seed(42)

        torch_model = BikeNet(n_features, hidden=hidden)
        keras_model = build_keras_model(n_features, hidden=hidden)

        torch_count = sum(p.numel() for p in torch_model.parameters())
        keras_count = keras_model.count_params()

        print("PyTorch parameters:", torch_count)
        print("Keras parameters:", keras_count)

        assert torch_count == keras_count

        # Transfer all three layers
        copy_torch_weights(torch_model, keras_model)

        # Check all validation observations
        difference = max_prediction_difference(torch_model, keras_model, x_val)

        print(f"Maximum prediction difference: {difference:.8f}")
        print("Equivalence check: PASS")
