# Software Solution

Monorepo for the Carboplatin Sensing AI platform. The backend and frontend are independent
applications; datasets, model implementations and notebooks are shared from this directory.

## Layout

```text
software-solution/
├── pyproject.toml            # uv workspace root
├── uv.lock                   # Workspace lockfile
├── .python-version
├── backend/                  # FastAPI + Python package
│   ├── README.md
│   ├── pyproject.toml
│   └── src/
│       └── software_solution/
├── cnn_transformer_implementare/
│   ├── README.md
│   ├── requirements.txt
│   ├── dataset.py
│   ├── model.py
│   ├── training.py
│   ├── plots.py
│   ├── inference.py
│   └── train.py
├── frontend/                 # React dashboard
│   ├── README.md
│   ├── package.json
│   ├── public/
│   └── src/
├── data/
│   ├── dataset_carboplatin_150_reale.csv
│   ├── sample_voltammetry.csv
│   └── test_samples/
├── data_extraction.ipynb
├── eda.ipynb
└── notebooks/
    └── exploration.ipynb
```

## Quick Start

Prerequisites: Python 3.15+, [uv](https://docs.astral.sh/uv/), Node.js 18+ and npm.

### Start both services

From `software-solution/`:

```bash
uv run start
```

This starts the API at `http://127.0.0.1:8000` and the dashboard at
`http://localhost:3000`. Press `Ctrl+C` to stop both. `dev` works as an alias for `start`.

On Windows, use `uv run start` (a bare `start` is a shell built-in).

### Backend only

```bash
cd backend
uv sync
uv run software-solution
```

The API is available at `http://127.0.0.1:8000`, with interactive documentation at
`http://127.0.0.1:8000/docs`.

### Frontend only

```bash
cd frontend
npm install
npm start
```

The dashboard runs at `http://localhost:3000`. It connects to the backend URL configured through
`REACT_APP_API_URL` in `frontend/.env` (see `frontend/.env.example`).

### Use the trained CNN in the web app

Train the CNN-Transformer once in an environment that has `torch`, then start the app.
The backend reuses any of these interpreters, so a single shared environment is enough:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r software-solution/requirements.txt
cd software-solution/cnn_transformer_implementare && python train.py
cd ../..
uv run start
```

Then select **CNN-Transformer** in the UI and upload `data/sample_voltammetry.csv`. The backend
detects the CNN environment automatically; see the
[CNN-Transformer README](cnn_transformer_implementare/README.md) for overrides.

## Documentation

- [Backend setup and API](backend/README.md)
- [Frontend setup and usage](frontend/README.md)
- [CNN-Transformer implementation](cnn_transformer_implementare/README.md)
- [EDA notebook](eda.ipynb)
- [Technical report](../documentation/technical_report.pdf)
