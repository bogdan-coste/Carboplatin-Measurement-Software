# Carboplatin Sensing Frontend

React dashboard for uploading voltammetry data and visualizing the prediction returned by the
FastAPI backend.

## Tech Stack

- React 19 (Create React App / `react-scripts` 5)
- Recharts for the voltammogram chart
- Lucide React for icons

## Project Layout

```text
frontend/
├── public/
│   ├── index.html
│   ├── manifest.json
│   └── ...
├── src/
│   ├── App.js            # Upload flow, results dashboard and chart
│   ├── App.css           # Application styles
│   ├── App.test.js       # UI smoke test
│   ├── index.js          # React entry point
│   ├── index.css         # Global styles
│   ├── reportWebVitals.js
│   └── setupTests.js
├── .env.example
├── package.json
└── README.md
```

## Setup

From `software-solution/frontend/`:

```bash
npm install
npm start
```

The app opens at `http://localhost:3000`. The backend must be running for analysis requests to
succeed.

To start the backend and frontend together, run this from `software-solution/`:

```bash
uv run start
```

## Configuration

Copy `.env.example` to `.env` and adjust the API URL if needed:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `REACT_APP_API_URL` | `http://127.0.0.1:8000` | Base URL of the FastAPI backend |

## Model Architectures

The upload panel lets the user choose between three model architectures:

| Option | Description |
| --- | --- |
| PINN-CV | Physics-informed network with Fick's diffusion model |
| CNN-Transformer | Convolutional feature extraction with self-attention |
| ANN / MLP | Dense neural baseline |

The selection is sent as the optional `architecture` form field. `cnn-transformer` uses the
trained model bundle when available; `pinn-cv` and `ann-mlp` currently return mock metrics.
Selecting a different architecture resets any displayed result so the UI never labels stale
results with the wrong model.

If the CNN bundle or its environment is missing, the backend returns HTTP 503 and the UI shows
the setup message inline. `physics_residual_loss` is `null` for CNN-Transformer because it does
not use the PINN physics penalty.

## API Contract

The frontend sends `POST /predict` as `multipart/form-data` with:

- `file` — a `.csv` or `.txt` file
- `architecture` — the selected model id (`pinn-cv`, `cnn-transformer` or `ann-mlp`)

The expected JSON response contains:

- `predicted_concentration_uM`
- `confidence_r2`
- `physics_residual_loss` (may be `null` for CNN-Transformer)
- `model_architecture`
- `preprocessing_pipeline`
- `potential_data` and `current_data` arrays

See [`../backend/README.md`](../backend/README.md) for backend setup.

## Scripts

| Command | Description |
| --- | --- |
| `npm start` | Start the development server |
| `npm run build` | Create a production build in `build/` |
| `npm test` | Run the test suite in watch mode |
| `CI=true npm test -- --watchAll=false` | Run the test suite once (CI) |
