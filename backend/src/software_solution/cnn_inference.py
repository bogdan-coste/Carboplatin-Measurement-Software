"""Adapter that runs CNN-Transformer inference in the model's own environment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
SOLUTION_ROOT = BACKEND_ROOT.parent
REPO_ROOT = SOLUTION_ROOT.parent
CNN_DIR = SOLUTION_ROOT / "cnn_transformer_implementare"
INFERENCE_SCRIPT = CNN_DIR / "inference.py"
DEFAULT_BUNDLE = CNN_DIR / "artifacts" / "inference_bundle.pt"

# Virtualenvs worth probing, most specific first. The CNN directory is the
# documented location; the repo root is where the shared environment usually
# lives; SOLUTION_ROOT/.venv is the backend's own environment.
VENV_DIRS = (CNN_DIR / ".venv", REPO_ROOT / ".venv", SOLUTION_ROOT / ".venv")


class CnnModelUnavailable(RuntimeError):
    """Raised when the CNN artifacts or its Python environment are missing."""


def _bundle_path() -> Path:
    return Path(os.environ.get("CNN_BUNDLE_PATH", DEFAULT_BUNDLE)).expanduser().resolve()


def _venv_python(venv_dir: Path) -> Path:
    """Return the interpreter path inside a virtualenv directory."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _has_torch(python: Path) -> bool:
    """Check whether an interpreter can import torch."""
    try:
        probe = subprocess.run(
            [str(python), "-c", "import torch"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _venv_python_candidates() -> list[Path]:
    return [_venv_python(venv_dir) for venv_dir in VENV_DIRS]


@lru_cache(maxsize=1)
def _python_executable() -> Path:
    """Locate a Python interpreter that can run the CNN inference script."""
    override = os.environ.get("CNN_INFERENCE_PYTHON")
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_file():
            return path
        raise CnnModelUnavailable(
            f"CNN_INFERENCE_PYTHON points to a missing file: {path}"
        )

    candidates = [*_venv_python_candidates(), Path(sys.executable)]
    for candidate in candidates:
        if candidate.is_file() and _has_torch(candidate):
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise CnnModelUnavailable(
        "CNN inference environment not found: no Python interpreter with "
        f"torch among {searched}. Install torch in one of those environments "
        "(see software-solution/requirements.txt) or set "
        "CNN_INFERENCE_PYTHON to a Python executable with torch installed."
    )


def run_cnn_inference(contents: bytes, filename: str) -> dict:
    """Execute the CNN inference script and return its JSON payload."""
    bundle_path = _bundle_path()
    if not bundle_path.is_file():
        raise CnnModelUnavailable(
            "CNN inference bundle not found. Run train.py in "
            "software-solution/cnn_transformer_implementare first."
        )

    if not INFERENCE_SCRIPT.is_file():
        raise CnnModelUnavailable(f"CNN inference script not found: {INFERENCE_SCRIPT}")

    python = _python_executable()
    suffix = Path(filename or "upload.csv").suffix or ".csv"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary.write(contents)
            temporary.flush()
            temporary_path = Path(temporary.name)

        result = subprocess.run(
            [
                str(python),
                str(INFERENCE_SCRIPT),
                "--input",
                str(temporary_path),
                "--bundle",
                str(bundle_path),
            ],
            cwd=CNN_DIR,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or "CNN inference process failed.")

    output_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("CNN inference returned no output.")

    try:
        payload = json.loads(output_lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Could not parse CNN inference output: {output_lines[-1]}"
        ) from error

    required = {
        "predicted_concentration_uM",
        "confidence_r2",
        "model_architecture",
        "preprocessing_pipeline",
        "potential_data",
        "current_data",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"CNN inference response is missing: {sorted(missing)}")

    return payload
