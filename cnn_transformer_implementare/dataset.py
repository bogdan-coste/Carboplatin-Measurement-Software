from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
SOLUTION_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET = SOLUTION_ROOT / "data" / "dataset_carboplatin_150_reale.csv"


def load_dataset(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected "
            "software-solution/data/dataset_carboplatin_150_reale.csv."
        )

    df = pd.read_csv(path)

    required = {"sample_id", "concentration_uM"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    potential_columns = sorted((column for column in df.columns if column.startswith("V_")),key=lambda column: float(column[2:]))

    if not potential_columns:
        raise ValueError("Dataset does not contain any V_* potential columns.")

    potentials = np.array([float(column[2:]) for column in potential_columns], dtype=np.float32)

    features = df[potential_columns].to_numpy(dtype=np.float32)

    concentrations = df["concentration_uM"].to_numpy(dtype=np.float32)

    if np.isnan(features).any() or np.isnan(concentrations).any():
        raise ValueError("Dataset contains missing values.")

    return df, potentials, features, concentrations

def extract_peak_features(features: np.ndarray, potentials: np.ndarray) -> np.ndarray:

    peak_region = (potentials > 0.8) & (potentials < 1.3)
    peak_heights = features[:, peak_region].max(axis=1)
    baseline = 0.5 * (
        features[:, potentials < 0.4].mean(axis=1)
        + features[:, potentials > 1.35].mean(axis=1)
    )
    return peak_heights - baseline

def split_dataset(
    concentrations: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    indices = np.arange(len(concentrations))

    bins = np.asarray(
        pd.qcut(concentrations, q=5, labels=False, duplicates="drop"),
        dtype=np.int64,
    )

    try:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.30, random_state=seed, stratify=bins
        )

        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=seed, stratify=bins[np.asarray(temp_idx, dtype=np.int64)]
        )

    except ValueError:
        train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=seed)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed)

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )

def prepare_tensors(
    features: np.ndarray,
    concentrations: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict, StandardScaler, StandardScaler]:
    """
        Normalize features and log-transform targets, then build PyTorch tensors.
    """

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = np.asarray(x_scaler.fit_transform(features[train_idx]), dtype=np.float64)
    x_val = np.asarray(x_scaler.transform(features[val_idx]), dtype=np.float64)
    x_test = np.asarray(x_scaler.transform(features[test_idx]), dtype=np.float64)

    y_train = np.ravel(np.asarray(y_scaler.fit_transform(np.log(concentrations[train_idx]).reshape(-1, 1)),dtype=np.float64))

    y_val = np.ravel(
        np.asarray(
            y_scaler.transform(np.log(concentrations[val_idx]).reshape(-1, 1)),
            dtype=np.float64,
        )
    )

    y_test = np.ravel(
        np.asarray(
            y_scaler.transform(np.log(concentrations[test_idx]).reshape(-1, 1)),
            dtype=np.float64,
        )
    )

    def to_tensor(values: np.ndarray, targets: np.ndarray):
        return (
            torch.tensor(values, dtype=torch.float32).unsqueeze(1),
            torch.tensor(targets, dtype=torch.float32),
        )

    data = {
        "train": to_tensor(x_train, y_train),
        "val": to_tensor(x_val, y_val),
        "test": to_tensor(x_test, y_test),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    return data, x_scaler, y_scaler
