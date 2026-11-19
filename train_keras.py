"""Lab 08: train the Keras bike demand model and save reports."""

import argparse
from pathlib import Path

import keras
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras_model import build_keras_model, get_keras_arrays
from prepare import prepare_data

SEED = 42
SPLIT_SEED = 42
HIDDEN = 32
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 80

REPORTS = Path(__file__).resolve().parent / "reports"


# ==========================================================
# 1. TRAIN KERAS MODEL
# ==========================================================


def train_keras_model(
    data,
    *,
    seed=SEED,
    hidden=HIDDEN,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    checkpoint_path=None,
    verbose=2,
):
    """Train with the same data and settings as Lab 07."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    # Seed the framework before creating the model
    keras.utils.set_random_seed(seed)

    # Use the SAME prepared data as PyTorch
    x_train, y_train = get_keras_arrays(data, "train")
    x_val, y_val = get_keras_arrays(data, "val")

    n_features = x_train.shape[1]

    # Build the same 8 -> hidden -> hidden -> 1 network
    model = build_keras_model(
        n_features=n_features,
        hidden=hidden,
    )

    # Configure training: Adam, MSE and MAE
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            epsilon=1e-8,
        ),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        jit_compile=False,
    )

    # Keep the best validation-loss weights
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=epochs,
            restore_best_weights=True,
        )
    ]

    # Save the best model, if a path is provided
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=False,
            )
        )

    # Target shape: (N, 1), matching Keras output
    y_train_2d = y_train.reshape(-1, 1)
    y_val_2d = y_val.reshape(-1, 1)

    # Keras handles the training loop
    result = model.fit(
        x_train,
        y_train_2d,
        validation_data=(x_val, y_val_2d),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )

    # Convert training results into a DataFrame
    history = pd.DataFrame(result.history)

    history = history.rename(
        columns={
            "loss": "train_loss",
            "mae": "train_mae",
        }
    )

    history.insert(
        0,
        "epoch",
        np.arange(1, len(history) + 1),
    )

    # Lowest validation MSE determines the best epoch
    best_row = history["val_loss"].idxmin()
    best_epoch = int(history.loc[best_row, "epoch"])

    return model, history, best_epoch


# ==========================================================
# 2. SAVE LEARNING CURVES
# ==========================================================


def save_learning_curves(history, best_epoch, path):
    """Plot training and validation MSE for each epoch."""

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        history["epoch"],
        history["train_loss"],
        label="Keras training MSE",
    )

    ax.plot(
        history["epoch"],
        history["val_loss"],
        label="Keras validation MSE",
    )

    ax.axvline(
        best_epoch,
        linestyle="--",
        color="gray",
        label=f"Best epoch: {best_epoch}",
    )

    ax.set_title("Keras Bike Demand Learning Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ==========================================================
# 3. MAIN PROGRAM
# ==========================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    # Same prepared rows as the PyTorch version
    data = prepare_data(seed=SPLIT_SEED)

    print("Training rows:", len(data["train"]))
    print("Validation rows:", len(data["val"]))
    print("Input features:", data["train"].x.shape[1])
    print("Keras backend:", keras.backend.backend())

    checkpoint = REPORTS / "keras_best.keras"

    # Run Keras training
    model, history, best_epoch = train_keras_model(
        data,
        seed=SEED,
        hidden=HIDDEN,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=args.epochs,
        checkpoint_path=checkpoint,
    )

    # Save one history row for every epoch
    history.to_csv(
        REPORTS / "history_keras.csv",
        index=False,
    )

    # Generate the main learning-curve PNG
    save_learning_curves(
        history,
        best_epoch,
        REPORTS / "learning_curves_keras.png",
    )

    best_row = history.loc[history["epoch"] == best_epoch].iloc[0]

    print("\nKERAS TRAINING SUMMARY")
    print("Parameters:", model.count_params())
    print("Epochs completed:", len(history))
    print("Best epoch:", best_epoch)
    print(
        "Best validation MSE:",
        round(float(best_row["val_loss"]), 4),
    )
    print(
        "Validation MAE at best epoch:",
        round(float(best_row["val_mae"]), 4),
    )

    # Verify that saved and restored models agree
    saved_model = keras.models.load_model(checkpoint)
    x_val, _ = get_keras_arrays(data, "val")

    live_pred = keras.ops.convert_to_numpy(model(x_val, training=False)).reshape(-1)

    saved_pred = keras.ops.convert_to_numpy(saved_model(x_val, training=False)).reshape(
        -1
    )

    np.testing.assert_allclose(
        live_pred,
        saved_pred,
        atol=1e-5,
        rtol=1e-5,
    )

    print("Saved checkpoint verification: PASS")
    print("Reports saved to:", REPORTS)
    print("No final test-set evaluation in Part 3.")


if __name__ == "__main__":
    main()
