from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_eda(
    potentials: np.ndarray,
    features: np.ndarray,
    concentrations: np.ndarray,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for index in np.argsort(concentrations)[::15]:
        axes[0].plot(
            potentials,
            features[index],
            alpha=0.7,
            label=f"{concentrations[index]:.0f} uM",
        )
    axes[0].set_xlabel("Potential (V)")
    axes[0].set_ylabel("Current (uA)")
    axes[0].set_title("Voltammograms at different concentrations")
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].hist(concentrations, bins=25, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Concentration (uM)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Concentration distribution")

    fig.tight_layout()
    fig.savefig(output_dir / "fig1_eda.png", dpi=120)
    plt.close(fig)


def plot_peak_calibration(
    concentrations: np.ndarray, peak_heights: np.ndarray, output_dir: Path
) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(concentrations, peak_heights, alpha=0.6, s=30, edgecolor="k")
    axis.set_xlabel("Concentration (uM)")
    axis.set_ylabel("Corrected peak height (uA)")
    axis.set_title("Peak height vs concentration")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_peak_vs_conc.png", dpi=120)
    plt.close(fig)


def plot_results(
    history: dict[str, list[float]],
    truth: np.ndarray,
    predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(history["train"], label="Train")
    axes[0].plot(history["val"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber loss")
    axes[0].set_title("Training curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        truth, predictions, alpha=0.75, s=55, edgecolor="k", label="CNN-Transformer"
    )
    axes[1].scatter(
        truth,
        baseline_predictions,
        alpha=0.55,
        s=55,
        marker="s",
        edgecolor="k",
        label="Baseline (traditional)",
    )
    upper = max(truth.max(), predictions.max(), baseline_predictions.max()) + 50
    lower = min(truth.min(), 0) - 50
    axes[1].plot([lower, upper], [lower, upper], "r--", label="Perfect")
    axes[1].set_xlabel("True concentration (uM)")
    axes[1].set_ylabel("Predicted concentration (uM)")
    axes[1].set_title("Test set predictions")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "fig3_results.png", dpi=120)
    plt.close(fig)
