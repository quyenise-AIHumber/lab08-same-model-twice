"""Lab 07: train PyTorch networks, compare baselines, save reproducible reports.

Requires the Part 1-3 files: bikes_data.py, prepare.py, and model.py.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch import nn

from model import BikeNet
from prepare import make_loaders, prepare_data

plt.switch_backend("Agg")  # Save plots without opening windows.

SEED = 42
REPORTS = Path(__file__).resolve().parent / "reports"


# ============================================================
# 1. TRAIN OR VALIDATE FOR ONE EPOCH
# ============================================================


def run_epoch(model, loader, loss_fn, optimizer=None):
    """Train for one epoch if an optimizer is supplied; otherwise only measure."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_abs_error = 0.0
    count = 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            predictions = model(x)
            assert predictions.shape == y.shape
            loss = loss_fn(predictions, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            size = len(y)
            total_loss += loss.item() * size
            total_abs_error += (predictions - y).abs().sum().item()
            count += size

    return total_loss / count, total_abs_error / count


# ============================================================
# 2. VERIFY THAT THE NETWORK CAN LEARN
# ============================================================


def check_small_batch(dataset, n_features):
    """Confirm that the network and optimizer can learn a tiny set of rows."""
    torch.manual_seed(123)
    model = BikeNet(n_features, hidden=64)
    x, y = dataset.x[:8], dataset.y[:8]
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    with torch.no_grad():
        initial = loss_fn(model(x), y).item()

    for _ in range(600):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final = loss_fn(model(x), y).item()

    print(f"Small-batch MSE: {initial:.4f} -> {final:.4f}")
    assert final < initial * 0.1, "Small-batch learning check failed."


# ============================================================
# 3. TRAIN THE MODEL FOR MULTIPLE EPOCHS
# ============================================================


def train_model(model, train_loader, val_loader, epochs, learning_rate):
    """Log every epoch and restore weights from lowest validation MSE."""
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_epoch = None
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, loss_fn, optimizer)

        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )

        # Save the weights from the lowest validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }

        if epoch in (1, epochs) or epoch % 20 == 0:
            print(
                f"  Epoch {epoch:3d}: train MSE={train_loss:8.3f}, "
                f"val MSE={val_loss:8.3f}, val MAE={val_mae:6.3f}"
            )

    model.load_state_dict(best_state)

    return pd.DataFrame(history), best_epoch


# ============================================================
# 4. GENERATE LEARNING CURVES
# ============================================================


def save_curves(history, best_epoch, destination, title):
    """Save training/validation MSE against epoch and mark best epoch."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        history["epoch"],
        history["train_loss"],
        label="Train MSE",
    )

    ax.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation MSE",
    )

    ax.axvline(
        best_epoch,
        color="gray",
        linestyle="--",
        label=f"Best epoch: {best_epoch}",
    )

    ax.set(
        title=title,
        xlabel="Epoch",
        ylabel="Mean squared error (bikes squared)",
    )

    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


# ============================================================
# 5. GENERATE PREDICTIONS AND CALCULATE MAE
# ============================================================


def predict_rentals(model, dataset):
    """Get predictions in rental units without computing gradients."""
    model.eval()

    with torch.no_grad():
        return model(dataset.x).cpu().numpy()


def mae(truth, predictions):
    return float(mean_absolute_error(truth, predictions))


# ============================================================
# 6. MAIN TRAINING AND EVALUATION PROCESS
# ============================================================


def main():
    torch.set_num_threads(1)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Load previously prepared data
    data = prepare_data(seed=SEED)
    n_features = data["train"].x.shape[1]

    print(
        "Rows:",
        len(data["train"]),
        len(data["val"]),
        len(data["test"]),
    )

    print("Input features:", n_features)

    # Confirm the neural network can learn
    check_small_batch(data["train"], n_features)

    # --------------------------------------------------------
    # 7. DEFINE MAIN RUN AND TWO EXPERIMENTS
    # --------------------------------------------------------

    configs = [
        {
            "name": "main",
            "hidden": 32,
            "learning_rate": 0.001,
            "epochs": 80,
        },
        {
            "name": "low_lr",
            "hidden": 32,
            "learning_rate": 0.0001,
            "epochs": 80,
        },
        {
            "name": "wide_net",
            "hidden": 128,
            "learning_rate": 0.001,
            "epochs": 80,
        },
    ]

    models = {}
    experiment_rows = []
    histories = {}

    # --------------------------------------------------------
    # 8. RUN ALL THREE EXPERIMENTS
    # --------------------------------------------------------

    for config in configs:
        name = config["name"]

        print(f"\nRunning {name}...")

        # Reset seeds for repeatable comparisons
        torch.manual_seed(SEED)

        train_loader, val_loader, _ = make_loaders(data, seed=SEED)

        model = BikeNet(
            n_features,
            hidden=config["hidden"],
        )

        history, best_epoch = train_model(
            model,
            train_loader,
            val_loader,
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
        )

        best_row = history.loc[history["epoch"] == best_epoch].iloc[0]

        experiment_rows.append(
            {
                **config,
                "best_epoch_by_val_loss": int(best_epoch),
                "best_validation_loss": float(best_row["val_loss"]),
                "val_mae_at_best_epoch": float(best_row["val_mae"]),
                "lowest_val_mae_seen": float(history["val_mae"].min()),
                "final_train_loss": float(history.iloc[-1]["train_loss"]),
                "final_val_loss": float(history.iloc[-1]["val_loss"]),
            }
        )

        # Generate learning curves
        filename = (
            "learning_curves.png" if name == "main" else f"learning_curves_{name}.png"
        )

        save_curves(
            history,
            best_epoch,
            REPORTS / filename,
            name,
        )

        models[name] = model
        histories[name] = history

    # --------------------------------------------------------
    # 9. SAVE TRAINING HISTORY AND EXPERIMENT RESULTS
    # --------------------------------------------------------

    histories["main"].to_csv(
        REPORTS / "history.csv",
        index=False,
    )

    pd.DataFrame(experiment_rows).to_csv(
        REPORTS / "experiments.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 10. SELECT BEST MODEL USING VALIDATION ONLY
    # --------------------------------------------------------

    winner = min(
        experiment_rows,
        key=lambda row: row["val_mae_at_best_epoch"],
    )

    selected = models[winner["name"]]

    print(
        "\nSelected using validation only:",
        winner["name"],
    )

    # --------------------------------------------------------
    # 11. BUILD BASELINE MODELS
    # --------------------------------------------------------

    x_train = data["train"].x.numpy()
    y_train = data["train"].y.numpy()

    x_val = data["val"].x.numpy()
    y_val = data["val"].y.numpy()

    # Baseline 1: Predict the training mean
    training_mean = float(y_train.mean())

    # Baseline 2: Linear regression
    linear_model = LinearRegression().fit(x_train, y_train)

    # Validation comparisons
    baseline_val = {
        "training_mean": mae(
            y_val,
            np.full_like(y_val, training_mean),
        ),
        "linear_regression": mae(
            y_val,
            linear_model.predict(x_val),
        ),
    }

    network_val = mae(
        y_val,
        predict_rentals(selected, data["val"]),
    )

    # --------------------------------------------------------
    # 12. FINAL TEST EVALUATION
    # --------------------------------------------------------
    # Test data are evaluated only after model selection.

    x_test = data["test"].x.numpy()
    y_test = data["test"].y.numpy()

    baseline_test = {
        "training_mean": mae(
            y_test,
            np.full_like(y_test, training_mean),
        ),
        "linear_regression": mae(
            y_test,
            linear_model.predict(x_test),
        ),
    }

    network_test = mae(
        y_test,
        predict_rentals(selected, data["test"]),
    )

    # --------------------------------------------------------
    # 13. SAVE FINAL METRICS
    # --------------------------------------------------------

    metrics = {
        "selected_experiment": winner["name"],
        "selection_rule": ("lowest validation MAE at best validation-loss epoch"),
        "split_sizes": {
            "train": len(y_train),
            "val": len(y_val),
            "test": len(y_test),
        },
        "units": "bikes rented (MAE)",
        "training_mean": {
            "validation_mae": baseline_val["training_mean"],
            "test_mae": baseline_test["training_mean"],
        },
        "linear_regression": {
            "validation_mae": baseline_val["linear_regression"],
            "test_mae": baseline_test["linear_regression"],
        },
        "network": {
            "validation_mae": network_val,
            "test_mae": network_test,
            "best_epoch_by_val_loss": winner["best_epoch_by_val_loss"],
            "best_validation_loss": winner["best_validation_loss"],
        },
    }

    with (REPORTS / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # --------------------------------------------------------
    # 14. PRINT FINAL RESULTS
    # --------------------------------------------------------

    print("\nFINAL MAE (bikes rented)")
    print(f"{'Model':<20} {'Validation':>12} {'Test':>12}")

    for name in (
        "training_mean",
        "linear_regression",
        "network",
    ):
        print(
            f"{name:<20} "
            f"{metrics[name]['validation_mae']:12.3f} "
            f"{metrics[name]['test_mae']:12.3f}"
        )

    print("\nSaved all reports to:", REPORTS)


if __name__ == "__main__":
    main()
