from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from model import CNNTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BUNDLE = SCRIPT_DIR / "artifacts" / "inference_bundle.pt"


def load_bundle(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(
            f"Inference bundle not found at {path}. Run train.py first."
        )

    try:
        bundle = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location=device)

    model = CNNTransformer(**bundle["model_config"])
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)
    model.eval()
    return bundle, model


def read_signal(path: Path, expected_potentials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)

    potential_columns = [column for column in df.columns if column.startswith("V_")]

    if potential_columns:
        if len(df) != 1:
            raise ValueError("Wide input files must contain exactly one sample row.")

        potentials = np.array([float(column[2:]) for column in potential_columns], dtype=np.float64)
        currents = np.asarray(
            pd.to_numeric(df[potential_columns].iloc[0], errors="coerce"),
            dtype=np.float64,
        )

    else:

        # check if the number of columns is 2
        if df.shape[1] < 2:
            raise ValueError("The uploaded file must contain at least two columns.")

        # extract column names from the data frame
        names = [str(column).lower() for column in df.columns]

        # find the index of the column for 'potential' in the column set
        # if the required column is not found, put 0
        potential_index = next(
            (
                index
                for index, name in enumerate(names)
                if "potential" in name or name in {"v", "voltage"}
            ),
            0,
        )

        # find the index of the column for 'current' in the column set
        # if the required column is not found, put 1 instead
        current_index = next(
            (
                index
                for index, name in enumerate(names)
                if "current" in name or name in {"i", "ua", "µa", "μa"}
            ),
            1,
        )

        if potential_index == current_index:
            potential_index, current_index = 0, 1

        potentials = np.asarray(pd.to_numeric(df.iloc[:, potential_index], errors="coerce"),dtype=np.float64)
        currents = np.asarray(pd.to_numeric(df.iloc[:, current_index], errors="coerce"), dtype=np.float64)

        valid = np.isfinite(potentials) & np.isfinite(currents)
        potentials = potentials[valid]
        currents = currents[valid]
        if len(potentials) < 2:
            raise ValueError("The uploaded file must contain at least two valid points.")

    order = np.argsort(potentials)
    potentials = potentials[order]
    currents = currents[order]

    if len(potentials) == len(expected_potentials) and np.allclose(
        potentials, expected_potentials, atol=1e-4, rtol=0
    ):
        aligned_currents = currents
    else:
        if (
            potentials.min() > expected_potentials.min() + 1e-6
            or potentials.max() < expected_potentials.max() - 1e-6
        ):
            raise ValueError(
                "Uploaded potential range does not cover the trained "
                f"{expected_potentials.min():.3f}-{expected_potentials.max():.3f} V window."
            )
        aligned_currents = np.interp(expected_potentials, potentials, currents)

    if not np.isfinite(aligned_currents).all():
        raise ValueError("Uploaded signal contains non-finite values.")

    return expected_potentials, aligned_currents


def predict(
    bundle: dict,
    model: CNNTransformer,
    potentials: np.ndarray,
    currents: np.ndarray,
    device: torch.device,
) -> dict:
    x_mean = np.asarray(bundle["x_scaler_mean"], dtype=np.float32)
    x_scale = np.asarray(bundle["x_scaler_scale"], dtype=np.float32)
    scaled = (currents.astype(np.float32) - x_mean) / x_scale

    tensor = torch.tensor(scaled, dtype=torch.float32).view(1, 1, -1).to(device)
    with torch.no_grad():
        scaled_prediction = float(model(tensor).item())

    y_mean = float(np.asarray(bundle["y_scaler_mean"]).reshape(-1)[0])
    y_scale = float(np.asarray(bundle["y_scaler_scale"]).reshape(-1)[0])
    log_concentration = scaled_prediction * y_scale + y_mean
    concentration = float(np.exp(log_concentration))

    metrics = bundle.get("metrics", {})
    return {
        "predicted_concentration_uM": concentration,
        "confidence_r2": float(metrics.get("R2", 0.0)),
        "mae_um": float(metrics.get("MAE", 0.0)),
        "rmse_um": float(metrics.get("RMSE", 0.0)),
        "mape_percent": float(metrics.get("MAPE", 0.0)),
        "model_architecture": bundle.get("architecture", "CNN-Transformer"),
        "preprocessing_pipeline": bundle.get(
            "preprocessing_pipeline", "StandardScaler + log target"
        ),
        "potential_data": potentials.tolist(),
        "current_data": currents.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CNN-Transformer inference.")
    parser.add_argument("--input", type=Path, required=True, help="CSV/TXT voltammetry file.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle, model = load_bundle(args.bundle, device)
    expected_potentials = np.asarray(bundle["potentials"], dtype=np.float32)
    potentials, currents = read_signal(args.input, expected_potentials)
    result = predict(bundle, model, potentials, currents, device)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
