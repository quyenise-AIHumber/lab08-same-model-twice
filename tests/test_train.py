import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import BikeNet
from prepare import make_loaders
from train import check_small_batch, run_epoch, train_model


def test_small_batch_can_be_memorized(prepared_data):
    check_small_batch(
        prepared_data["train"],
        n_features=8,
    )


def test_validation_does_not_update_weights(
    prepared_data,
):
    model = BikeNet(8, hidden=16)

    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    _, val_loader, _ = make_loaders(prepared_data)

    loss, error = run_epoch(model, val_loader, nn.MSELoss())

    assert np.isfinite(loss)
    assert np.isfinite(error)

    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())

    assert not model.training


def test_gradients_are_cleared_before_each_batch():
    torch.manual_seed(42)

    model = BikeNet(8, hidden=16)
    reference = copy.deepcopy(model)

    x = torch.ones((1, 8))
    y = torch.tensor([10.0])

    loader = DataLoader(
        TensorDataset(
            x.repeat(2, 1),
            y.repeat(2),
        ),
        batch_size=1,
    )

    # No weight changes: identical batches must
    # produce identical gradients after zero_grad().
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    run_epoch(model, loader, nn.MSELoss(), optimizer)

    nn.MSELoss()(reference(x), y).backward()

    for param, expected in zip(
        model.parameters(),
        reference.parameters(),
        strict=True,
    ):
        assert torch.allclose(
            param.grad,
            expected.grad,
            atol=1e-6,
        )


def test_epoch_uses_sample_weighted_average_for_partial_batches():
    class FirstColumn(nn.Module):
        def forward(self, x):
            return x[:, 0]

    features = torch.tensor(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    targets = torch.tensor([2.0, 4.0, 8.0])

    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=2,
    )

    mse, error = run_epoch(FirstColumn(), loader, nn.MSELoss())

    assert mse == pytest.approx(10.0)
    assert error == pytest.approx(8 / 3)


def test_training_history_and_best_epoch_are_restored():
    torch.manual_seed(10)

    model = BikeNet(8, hidden=8)

    features = torch.randn(32, 8)
    labels = features[:, 0] + features[:, 1]

    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=8,
    )

    history, best_epoch = train_model(
        model,
        loader,
        loader,
        epochs=8,
        learning_rate=0.01,
    )

    assert len(history) == 8

    required = {
        "epoch",
        "train_loss",
        "train_mae",
        "val_loss",
        "val_mae",
    }

    assert required <= set(history.columns)

    lowest_loss_epoch = int(
        history.loc[
            history["val_loss"].idxmin(),
            "epoch",
        ]
    )

    assert best_epoch == lowest_loss_epoch

    restored_loss, _ = run_epoch(model, loader, nn.MSELoss())

    assert restored_loss == pytest.approx(
        float(history["val_loss"].min()),
        abs=1e-5,
    )
