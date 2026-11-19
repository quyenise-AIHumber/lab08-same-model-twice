"""Tests for comparison orchestration and saved results."""

import sys
from pathlib import Path

import pandas as pd
import pytest

import compare


@pytest.mark.parametrize("smoke", [True, False])
def test_comparison_modes(
    monkeypatch,
    tmp_path,
    prepared_data,
    smoke,
):
    # Avoid six expensive training runs in a unit test.
    monkeypatch.setattr(
        compare,
        "prepare_data",
        lambda seed: prepared_data,
    )

    monkeypatch.setattr(compare, "REPORTS", tmp_path)

    monkeypatch.setattr(
        compare.keras.backend,
        "backend",
        lambda: "tensorflow",
    )

    monkeypatch.setattr(
        compare.torch,
        "set_num_threads",
        lambda n: None,
    )

    def fake_run(framework):
        def run(data, seed, epochs, score_test):
            return {
                "framework": framework,
                "seed": seed,
                "best_epoch": epochs,
                "validation_mae": 5.0,
                "test_mae": 4.0 if score_test else None,
                "seconds": 1.0,
            }

        return run

    monkeypatch.setattr(
        compare,
        "benchmark_pytorch",
        fake_run("PyTorch"),
    )

    monkeypatch.setattr(
        compare,
        "benchmark_keras",
        fake_run("Keras"),
    )

    argv = ["compare.py"]

    if smoke:
        argv.append("--smoke")

    monkeypatch.setattr(sys, "argv", argv)

    compare.main()

    filename = "comparison_smoke.csv" if smoke else "comparison.csv"

    result = pd.read_csv(tmp_path / filename)

    assert len(result) == (2 if smoke else 6)

    if smoke:
        assert result["test_mae"].isna().all()
    else:
        assert result["test_mae"].notna().all()
        assert set(result["seed"]) == {42, 43, 44}


def test_committed_comparison_results():
    project = Path(__file__).resolve().parents[1]
    path = project / "reports" / "comparison.csv"

    assert path.exists()

    result = pd.read_csv(path)

    required_columns = {
        "framework",
        "seed",
        "best_epoch",
        "validation_mae",
        "test_mae",
        "seconds",
    }

    assert required_columns <= set(result.columns)

    assert len(result) == 6

    for framework in ("PyTorch", "Keras"):
        rows = result[result["framework"] == framework]

        assert set(rows["seed"]) == {42, 43, 44}

    assert result["test_mae"].notna().all()
    assert (result["seconds"] > 0).all()
