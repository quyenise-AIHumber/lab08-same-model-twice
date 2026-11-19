import pytest
import torch
from torch import nn

from model import BikeNet


@pytest.mark.parametrize("hidden", [4, 16, 32])
def test_network_returns_one_prediction_per_row(hidden):
    model = BikeNet(n_features=8, hidden=hidden)

    predictions = model(torch.randn(5, 8))

    assert predictions.shape == (5,)

    layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]

    assert len(layers) == 3


def test_optimizer_step_changes_each_linear_layer():
    torch.manual_seed(42)

    model = BikeNet(8, hidden=16)
    x = torch.randn(64, 8)
    y = torch.arange(64, dtype=torch.float32) / 2

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]

    before = [layer.weight.detach().clone() for layer in layers]

    optimizer.zero_grad()

    loss = nn.MSELoss()(model(x), y)

    loss.backward()
    optimizer.step()

    assert all(
        not torch.equal(old, layer.weight)
        for old, layer in zip(before, layers, strict=True)
    )
