import numpy as np
import torch

from prepare import NUMERIC, BikeDataset, make_loaders


def test_dataset_length_shapes_and_float32():
    features = np.ones((5, 8), dtype=np.float64)
    targets = np.arange(5, dtype=np.float64)

    dataset = BikeDataset(features, targets)

    assert len(dataset) == 5

    x, y = dataset[0]

    assert x.shape == (8,)
    assert y.shape == ()
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_splits_have_correct_sizes_and_no_overlap(
    prepared_data,
):
    data = prepared_data

    assert (
        len(data["train"]),
        len(data["val"]),
        len(data["test"]),
    ) == (3600, 1200, 1200)

    train_ids = set(data["train_df"].index)
    val_ids = set(data["val_df"].index)
    test_ids = set(data["test_df"].index)

    assert len(train_ids | val_ids | test_ids) == 6000

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_scaler_fits_only_training_rows(prepared_data):
    data = prepared_data

    scaler = data["preprocessor"].named_transformers_["num"]

    original_train_means = data["train_df"][NUMERIC].mean().to_numpy()

    assert np.allclose(scaler.mean_, original_train_means)

    n_numeric = len(NUMERIC)

    train_means = data["train"].x[:, :n_numeric].mean(0).numpy()

    val_means = data["val"].x[:, :n_numeric].mean(0).numpy()

    assert np.allclose(train_means, 0.0, atol=1e-5)

    assert not np.allclose(val_means, 0.0, atol=1e-3)


def test_training_loader_shuffle_is_reproducible(
    prepared_data,
):
    train_a, val_a, test_a = make_loaders(prepared_data, seed=42)

    train_b, _, _ = make_loaders(prepared_data, seed=42)

    xa, ya = next(iter(train_a))
    xb, yb = next(iter(train_b))

    assert torch.equal(xa, xb)
    assert torch.equal(ya, yb)

    # The shuffled batch must differ from original order.
    assert not torch.equal(
        xa,
        prepared_data["train"].x[: len(xa)],
    )

    # Validation and test loaders must not shuffle.
    assert not isinstance(
        val_a.sampler,
        torch.utils.data.RandomSampler,
    )

    assert not isinstance(
        test_a.sampler,
        torch.utils.data.RandomSampler,
    )
