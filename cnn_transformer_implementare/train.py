from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from dataset import (
    DEFAULT_DATASET,
    extract_peak_features,
    load_dataset,
    prepare_tensors,
    split_dataset,
)
from model import CNNTransformer
from plots import plot_eda, plot_peak_calibration, plot_results
from sklearn.linear_model import LinearRegression
from training import evaluate, regression_metrics, train_model

SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN-Transformer for carboplatin quantification."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, potentials, features, concentrations = load_dataset(args.dataset)
    print(f"\nDataset shape: {features.shape}")
    print(f"Samples: {len(concentrations)}")
    print(f"Potential range: {potentials.min():.3f} - {potentials.max():.3f} V")
    print(
        f"Concentration range: {concentrations.min():.1f} - "
        f"{concentrations.max():.1f} uM"
    )

    train_idx, val_idx, test_idx = split_dataset(concentrations, args.seed)
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    data, x_scaler, y_scaler = prepare_tensors(
        features, concentrations, train_idx, val_idx, test_idx
    )
    train_data = data["train"]
    val_data = data["val"]
    test_data = data["test"]

    model_config = {
        "input_length": int(features.shape[1]),
        "cnn_channels": (16, 32, 64),
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.2,
    }
    model = CNNTransformer(**model_config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count:,}")

    best_model_path = args.output_dir / "best_model.pt"
    model, history = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        best_model_path=best_model_path,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )

    train_metrics, _ = evaluate(
        model, train_data[0], concentrations[train_idx], y_scaler, device, "Train"
    )
    val_metrics, _ = evaluate(
        model, val_data[0], concentrations[val_idx], y_scaler, device, "Validation"
    )
    test_metrics, test_predictions = evaluate(
        model, test_data[0], concentrations[test_idx], y_scaler, device, "Test"
    )

    peak_train = extract_peak_features(features[train_idx], potentials).reshape(-1, 1)
    peak_test = extract_peak_features(features[test_idx], potentials).reshape(-1, 1)
    baseline = LinearRegression().fit(peak_train, np.log(concentrations[train_idx]))
    baseline_predictions = np.exp(baseline.predict(peak_test))
    baseline_metrics = regression_metrics(concentrations[test_idx], baseline_predictions)

    print("\nBaseline (traditional calibration):")
    print(f"  MAE:  {baseline_metrics['MAE']:7.2f} uM")
    print(f"  RMSE: {baseline_metrics['RMSE']:7.2f} uM")
    print(f"  R2:   {baseline_metrics['R2']:7.4f}")

    residuals = np.abs(concentrations[test_idx] - test_predictions)
    print("\nBest test predictions:")
    for index in np.argsort(residuals)[:3]:
        print(
            f"  true={concentrations[test_idx][index]:7.1f} | "
            f"pred={test_predictions[index]:7.1f} | "
            f"error={residuals[index]:5.1f}"
        )

    print("\nWorst test predictions:")
    for index in np.argsort(residuals)[-3:]:
        print(
            f"  true={concentrations[test_idx][index]:7.1f} | "
            f"pred={test_predictions[index]:7.1f} | "
            f"error={residuals[index]:5.1f}"
        )

    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R2':>8}")
    print("=" * 60)
    print(
        f"{'Baseline (traditional)':<25} "
        f"{baseline_metrics['MAE']:>10.2f} "
        f"{baseline_metrics['RMSE']:>10.2f} "
        f"{baseline_metrics['R2']:>8.4f}"
    )
    print(
        f"{'CNN-Transformer (ours)':<25} "
        f"{test_metrics['MAE']:>10.2f} "
        f"{test_metrics['RMSE']:>10.2f} "
        f"{test_metrics['R2']:>8.4f}"
    )
    print("=" * 60)

    plot_eda(potentials, features, concentrations, args.output_dir)
    plot_peak_calibration(
        concentrations,
        extract_peak_features(features, potentials),
        args.output_dir,
    )
    plot_results(
        history,
        concentrations[test_idx],
        test_predictions,
        baseline_predictions,
        args.output_dir,
    )

    metrics_path = args.output_dir / "metrics.txt"
    metrics_path.write_text(
        "\n".join(
            [
                f"Train: {train_metrics}",
                f"Validation: {val_metrics}",
                f"Test: {test_metrics}",
                f"Baseline: {baseline_metrics}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inference_bundle = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "potentials": np.asarray(potentials, dtype=np.float32),
        "x_scaler_mean": np.asarray(x_scaler.mean_, dtype=np.float32),
        "x_scaler_scale": np.asarray(x_scaler.scale_, dtype=np.float32),
        "y_scaler_mean": np.asarray(y_scaler.mean_, dtype=np.float32),
        "y_scaler_scale": np.asarray(y_scaler.scale_, dtype=np.float32),
        "metrics": {key: float(value) for key, value in test_metrics.items()},
        "architecture": "CNN-Transformer",
        "preprocessing_pipeline": "StandardScaler + log target",
    }
    bundle_path = args.output_dir / "inference_bundle.pt"
    torch.save(inference_bundle, bundle_path)

    print(f"\nArtifacts saved to {args.output_dir}")
    print(f"Inference bundle: {bundle_path}")


if __name__ == "__main__":
    main()
