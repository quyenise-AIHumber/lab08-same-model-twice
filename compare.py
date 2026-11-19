"""Lab 08: compare PyTorch and Keras on three random seeds.

Normal: 6 models, 80 epochs each, final test evaluation.
--smoke: 2 models, 2 epochs each, no test evaluation.
"""

import argparse
from pathlib import Path
from time import perf_counter

import keras
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

from keras_model import get_keras_arrays
from model import BikeNet
from prepare import make_loaders, prepare_data
from train import train_model
from train_keras import train_keras_model

# ------------------------------------------------------------
# 1. FAIR COMPARISON SETTINGS
# ------------------------------------------------------------

SPLIT_SEED = 42
SEEDS = (42, 43, 44)
HIDDEN = 32
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 80

REPORTS = Path(__file__).resolve().parent / "reports"


# ------------------------------------------------------------
# 2. RUN ONE PYTORCH EXPERIMENT
# ------------------------------------------------------------


def benchmark_pytorch(data, seed, epochs, score_test):
    """Train PyTorch, restore best weights, record results."""

    torch.manual_seed(seed)

    start = perf_counter()

    train_loader, val_loader, _ = make_loaders(
        data,
        seed=seed,
        batch_size=BATCH_SIZE,
    )

    model = BikeNet(
        data["train"].x.shape[1],
        hidden=HIDDEN,
    )

    history, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=LEARNING_RATE,
    )

    seconds = perf_counter() - start

    # Validation MAE from the best validation-loss epoch
    best_row = history.loc[history["epoch"] == best_epoch].iloc[0]

    test_mae = None

    # Evaluate test data once after model selection
    if score_test:
        model.eval()

        with torch.no_grad():
            predictions = model(data["test"].x).cpu().numpy().reshape(-1)

        truth = data["test"].y.cpu().numpy().reshape(-1)

        test_mae = float(mean_absolute_error(truth, predictions))

    return {
        "framework": "PyTorch",
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_mae": float(best_row["val_mae"]),
        "test_mae": test_mae,
        "seconds": round(seconds, 3),
    }


# ------------------------------------------------------------
# 3. RUN ONE KERAS EXPERIMENT
# ------------------------------------------------------------


def benchmark_keras(data, seed, epochs, score_test):
    """Train Keras with the same configuration and data."""

    keras.backend.clear_session()

    start = perf_counter()

    model, history, best_epoch = train_keras_model(
        data,
        seed=seed,
        hidden=HIDDEN,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        checkpoint_path=None,
        verbose=0,
    )

    seconds = perf_counter() - start

    best_row = history.loc[history["epoch"] == best_epoch].iloc[0]

    test_mae = None

    if score_test:
        x_test, y_test = get_keras_arrays(data, "test")

        predictions = keras.ops.convert_to_numpy(model(x_test, training=False)).reshape(
            -1
        )

        test_mae = float(
            mean_absolute_error(
                y_test.reshape(-1),
                predictions,
            )
        )

    result = {
        "framework": "Keras",
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_mae": float(best_row["val_mae"]),
        "test_mae": test_mae,
        "seconds": round(seconds, 3),
    }

    keras.backend.clear_session()

    return result


# ------------------------------------------------------------
# 4. RUN THE FULL FRAMEWORK COMPARISON
# ------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run two short training checks without test scoring.",
    )

    args = parser.parse_args()

    if keras.backend.backend() != "tensorflow":
        raise RuntimeError("Lab 08 requires the TensorFlow Keras backend.")

    # Consistent CPU threading for the comparison
    torch.set_num_threads(1)

    REPORTS.mkdir(parents=True, exist_ok=True)

    # The SAME data split for every framework and seed
    data = prepare_data(seed=SPLIT_SEED)

    print(
        "Prepared rows:",
        len(data["train"]),
        len(data["val"]),
        len(data["test"]),
    )

    print("Input features:", data["train"].x.shape[1])
    print("Keras backend:", keras.backend.backend())

    if args.smoke:
        seeds = (42,)
        epochs = 2
    else:
        seeds = SEEDS
        epochs = EPOCHS

    score_test = not args.smoke
    rows = []

    # Train both frameworks for every selected seed
    for seed in seeds:
        print(f"\nSEED {seed} - PyTorch ({epochs} epochs)")

        pytorch_result = benchmark_pytorch(data, seed, epochs, score_test)
        rows.append(pytorch_result)

        print(f"SEED {seed} - Keras ({epochs} epochs)")

        keras_result = benchmark_keras(data, seed, epochs, score_test)
        rows.append(keras_result)

    # --------------------------------------------------------
    # 5. SAVE ALL RESULTS
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        rows,
        columns=[
            "framework",
            "seed",
            "best_epoch",
            "validation_mae",
            "test_mae",
            "seconds",
        ],
    )

    filename = "comparison_smoke.csv" if args.smoke else "comparison.csv"

    comparison.to_csv(
        REPORTS / filename,
        index=False,
    )

    print("\nPER-RUN RESULTS")
    print(comparison.round(3).to_string(index=False))

    # --------------------------------------------------------
    # 6. SUMMARIZE THE THREE-SEED COMPARISON
    # --------------------------------------------------------

    if not args.smoke:
        summary = comparison.groupby("framework").agg(
            runs=("seed", "size"),
            validation_mae_mean=("validation_mae", "mean"),
            validation_mae_std=("validation_mae", "std"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            seconds_mean=("seconds", "mean"),
        )

        print("\nSUMMARY BY FRAMEWORK")

        print(summary.round(3).to_string())

        assert len(comparison) == 6
        assert comparison["test_mae"].notna().all()

    else:
        print("\nSMOKE CHECK: Test data were NOT scored.")

    print("\nSaved:", REPORTS / filename)


if __name__ == "__main__":
    main()
