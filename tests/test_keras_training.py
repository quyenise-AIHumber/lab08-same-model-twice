"""Test Keras learning, checkpointing and reproducibility."""

import keras
import numpy as np

from keras_model import build_keras_model, get_keras_arrays
from train_keras import train_keras_model


# TEST 6: Keras can memorize a tiny batch
def test_keras_can_memorize_small_batch():
    keras.backend.clear_session()
    keras.utils.set_random_seed(17)

    x = np.eye(8, dtype=np.float32)
    y = np.arange(1, 9, dtype=np.float32).reshape(-1, 1)

    model = build_keras_model(8, hidden=32)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="mse",
        jit_compile=False,
    )

    initial = float(model.evaluate(x, y, verbose=0))

    model.fit(
        x,
        y,
        epochs=160,
        batch_size=8,
        shuffle=False,
        verbose=0,
    )

    final = float(model.evaluate(x, y, verbose=0))

    assert final < initial * 0.05


# TEST 7: Save and reload preserves predictions
def test_keras_save_reload(tmp_path):
    keras.utils.set_random_seed(42)

    model = build_keras_model(8, hidden=16)

    x = np.random.default_rng(42).normal(size=(10, 8)).astype(np.float32)

    before = keras.ops.convert_to_numpy(model(x, training=False))

    path = tmp_path / "roundtrip.keras"
    model.save(path)

    reloaded = keras.models.load_model(path)

    after = keras.ops.convert_to_numpy(reloaded(x, training=False))

    np.testing.assert_allclose(
        before,
        after,
        rtol=1e-6,
        atol=1e-6,
    )


# TEST 8: Restored weights reproduce best validation MSE
def test_best_validation_weights_are_restored(
    prepared_data,
):
    keras.backend.clear_session()

    model, history, best_epoch = train_keras_model(
        prepared_data,
        seed=42,
        hidden=16,
        learning_rate=0.001,
        batch_size=64,
        epochs=5,
        verbose=0,
    )

    x_val, y_val = get_keras_arrays(prepared_data, "val")

    predictions = keras.ops.convert_to_numpy(model(x_val, training=False)).reshape(-1)

    restored_mse = float(np.mean((predictions - y_val) ** 2))

    best_mse = float(history["val_loss"].min())

    assert best_epoch == int(history.loc[history["val_loss"].idxmin(), "epoch"])

    np.testing.assert_allclose(
        restored_mse,
        best_mse,
        rtol=1e-4,
        atol=1e-3,
    )


# TEST 9: Same seed produces repeatable short runs
def test_same_seed_repeats_short_training(
    prepared_data,
):
    def run_once():
        keras.backend.clear_session()

        model, history, _ = train_keras_model(
            prepared_data,
            seed=123,
            hidden=16,
            learning_rate=0.001,
            batch_size=64,
            epochs=3,
            verbose=0,
        )

        x_val, _ = get_keras_arrays(prepared_data, "val")

        predictions = keras.ops.convert_to_numpy(model(x_val, training=False)).reshape(
            -1
        )

        return history["val_loss"].to_numpy(), predictions

    loss_a, pred_a = run_once()
    loss_b, pred_b = run_once()

    np.testing.assert_allclose(loss_a, loss_b, rtol=1e-3, atol=1e-3)

    np.testing.assert_allclose(pred_a, pred_b, rtol=1e-3, atol=1e-3)
