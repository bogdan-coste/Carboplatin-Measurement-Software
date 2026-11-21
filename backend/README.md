# Backend — Carboplatin Sensing API

FastAPI backend for the Carboplatin Sensing AI platform. It accepts a voltammetry trace as a
CSV/TXT upload and returns a concentration estimate together with model diagnostics and the
processed signal.

## Project Layout

```text
backend/
├── src/
│   └── software_solution/
│       ├── __init__.py    # Package entry point (runs uvicorn)
│       ├── main.py        # FastAPI app and /predict endpoint
│       └── dev.py         # Combined backend + frontend launcher
├── pyproject.toml
└── README.md
```

Shared workspace files (`uv.lock`, `.python-version`, the workspace `pyproject.toml`) live in
`software-solution/`.

Shared datasets live in `../data/`.

## Requirements

- Python 3.15+ (pinned in the workspace `.python-version`)
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup and Run

### Start both services

From `software-solution/`:

```bash
uv run start
```

You can also run the same command from `software-solution/backend/`; uv discovers the workspace
root automatically.

This launches the API on `http://127.0.0.1:8000` and the React dev server on
`http://localhost:3000`. Use `Ctrl+C` to stop both. `dev` is available as an alias for `start`.

Optional flags:

| Flag | Purpose |
| --- | --- |
| `--backend-port 8000` | Change the API port |
| `--frontend-port 3000` | Change the React dev server port |
| `--no-reload` | Disable backend auto-reload |
| `--dry-run` | Print the commands without starting them |

> On Windows, type `uv run start` rather than a bare `start`, because `start` is a shell built-in.

### Start the backend only

From `software-solution/backend/`:

```bash
uv sync
uv run software-solution
```

The server starts on `http://127.0.0.1:8000`.

Alternatively, after activating the virtual environment:

```bash
uvicorn software_solution.main:app --reload
```

Interactive API documentation is served at `http://127.0.0.1:8000/docs`.

## API

### `GET /`

Health check.

```json
{
  "status": "online",
  "message": "Carboplatin Sensing API connected."
}
```

### `POST /predict`

Multipart upload with the following form fields:

- `file` — required `.csv` or `.txt` voltammetry file
- `architecture` — optional model id: `pinn-cv` (default), `cnn-transformer` or `ann-mlp`

For the mock architectures, the first two columns are interpreted as **potential (V)** and
**current (µA)**. The file must contain at least two columns.

Examples, from `software-solution/`:

```bash
# Mock PINN-CV response
curl -X POST -F "file=@data/sample_voltammetry.csv" \
  http://127.0.0.1:8000/predict

# Trained CNN-Transformer inference
curl -X POST -F "file=@data/sample_voltammetry.csv" \
  -F "architecture=cnn-transformer" \
  http://127.0.0.1:8000/predict
```

Example mock response:

```json
{
  "predicted_concentration_uM": 14.85,
  "physics_residual_loss": 0.0021,
  "confidence_r2": 0.984,
  "model_architecture": "PINN-CV (Fick's Diffusion)",
  "preprocessing_pipeline": "splearn Denoising + MinMax",
  "potential_data": [-0.6, -0.55],
  "current_data": [-1.2, -1.15]
}
```

CNN responses use the trained model metrics and set `physics_residual_loss` to `null` because
the CNN-Transformer does not use the PINN physics penalty.

## CNN Inference Setup

The FastAPI backend does not import PyTorch. CNN requests run `inference.py` from
`cnn_transformer_implementare/` in a separate virtual environment that has `torch` installed
(dependencies are listed in `requirements.txt` at the `software-solution/` root):

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r software-solution/requirements.txt
cd software-solution/cnn_transformer_implementare && python train.py
```

The backend probes these locations for an interpreter that can `import torch`, in order, and
uses the first match:

1. `CNN_INFERENCE_PYTHON`, if set
2. `software-solution/cnn_transformer_implementare/.venv`
3. `.venv` at the repository root
4. `software-solution/.venv`
5. the interpreter running the API

So a single shared environment at the repository root is enough; no `.venv` inside
`cnn_transformer_implementare/` is required. To override the search entirely, set these
variables before starting the API:

| Variable | Purpose |
| --- | --- |
| `CNN_INFERENCE_PYTHON` | Python executable containing `torch` |
| `CNN_BUNDLE_PATH` | Path to `inference_bundle.pt` |

If the bundle or environment is missing, `/predict` returns HTTP 503 with setup instructions.

## Data

- `../data/sample_voltammetry.csv` — two-column example input for the API/frontend demo
  (extracted from the first real sample).
- `../data/dataset_carboplatin_150_reale.csv` — real 150-sample dataset used by the model
  implementations.
- `../data/test_samples/` — held-out test traces exported as two-column CSVs with known
  concentrations.

## Development Notes

- `pinn-cv` and `ann-mlp` currently return fixed placeholder metrics. The `cnn-transformer`
  route uses the trained bundle when available.
- CORS is open to all origins for local development.
- There are no automated backend tests yet; the frontend test suite lives in
  `../frontend/src/App.test.js`.
