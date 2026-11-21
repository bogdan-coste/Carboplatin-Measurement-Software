import React, { useMemo, useRef, useState } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Activity,
  AlertCircle,
  BrainCircuit,
  Check,
  CheckCircle2,
  Cpu,
  Database,
  FileSpreadsheet,
  FlaskConical,
  Gauge,
  Layers,
  Loader2,
  Network,
  Sparkles,
  Target,
  TrendingDown,
  Upload,
  X,
} from 'lucide-react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';
const ACCEPTED_EXTENSIONS = ['.csv', '.txt'];

const MODEL_OPTIONS = [
  {
    id: 'pinn-cv',
    name: 'PINN-CV',
    tagline: 'Physics-informed',
    icon: BrainCircuit,
    description: "The PINN-CV network combines measured data with Fick's diffusion model.",
    tags: ['Physics-informed', "Fick's diffusion", 'Signal denoising'],
  },
  {
    id: 'cnn-transformer',
    name: 'CNN-Transformer',
    tagline: 'Attention-based',
    icon: Network,
    description:
      'Convolutional features are refined by transformer attention across the signal.',
    tags: ['Convolutional encoder', 'Self-attention', 'Sequence modelling'],
  },
  {
    id: 'ann-mlp',
    name: 'ANN / MLP',
    tagline: 'Dense baseline',
    icon: Layers,
    description:
      'A dense neural baseline maps the preprocessed trace to a concentration estimate.',
    tags: ['Dense layers', 'Fast inference', 'Baseline'],
  },
];

const DEFAULT_MODEL_ID = MODEL_OPTIONS[0].id;

const formatMetric = (value, digits = 3) => {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  if (value !== 0 && Math.abs(value) < 10 ** -digits) return value.toExponential(2);
  return value.toLocaleString('en-US', { maximumFractionDigits: digits });
};

const formatFileSize = (bytes) => {
  if (!Number.isFinite(bytes)) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const isSupportedFile = (candidate) =>
  ACCEPTED_EXTENSIONS.some((extension) => candidate.name.toLowerCase().endsWith(extension));

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div className="chart-tooltip">
      <span className="chart-tooltip__label">Potential {Number(label).toFixed(2)} V</span>
      <span className="chart-tooltip__value">{Number(payload[0].value).toFixed(2)} µA</span>
    </div>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(null);
  const [architecture, setArchitecture] = useState(DEFAULT_MODEL_ID);
  const inputRef = useRef(null);

  const selectedModel =
    MODEL_OPTIONS.find((option) => option.id === architecture) ?? MODEL_OPTIONS[0];

  const acceptFile = (candidate) => {
    if (!candidate) return;

    if (!isSupportedFile(candidate)) {
      setFile(null);
      setResults(null);
      setError('Unsupported file type. Please upload a .csv or .txt file.');
      return;
    }

    setError(null);
    setResults(null);
    setFile(candidate);
  };

  const handleFileChange = (event) => {
    acceptFile(event.target.files?.[0]);
    event.target.value = '';
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  };

  const handleArchitectureChange = (event) => {
    setArchitecture(event.target.value);
    setError(null);
    setResults(null);
  };

  const handleRemoveFile = () => {
    setFile(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  const handleReset = () => {
    setFile(null);
    setResults(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Select a CSV file before running the analysis.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('architecture', architecture);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let message = 'The server could not process this file.';
        try {
          const payload = await response.json();
          if (typeof payload?.detail === 'string') message = payload.detail;
        } catch {
          // Keep the fallback message when the response has no JSON body.
        }
        throw new Error(message);
      }

      setResults(await response.json());
    } catch (err) {
      const message =
        err instanceof TypeError
          ? `Could not reach the prediction API at ${API_URL}. Make sure the backend is running.`
          : err.message;
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const chartData = useMemo(() => {
    if (!results?.potential_data || !results?.current_data) return [];

    return results.potential_data.map((potential, index) => ({
      potential,
      current: results.current_data[index],
    }));
  }, [results]);

  const confidence = typeof results?.confidence_r2 === 'number' ? results.confidence_r2 : 0;
  const confidencePercent = Math.round(Math.min(Math.max(confidence, 0), 1) * 100);
  const hasPhysicsLoss = typeof results?.physics_residual_loss === 'number';

  const signalSummary = useMemo(() => {
    if (chartData.length === 0) return null;

    const potentials = chartData.map((point) => Number(point.potential)).filter(Number.isFinite);
    const currents = chartData.map((point) => Number(point.current)).filter(Number.isFinite);

    if (potentials.length === 0 || currents.length === 0) return null;

    return {
      minPotential: Math.min(...potentials),
      maxPotential: Math.max(...potentials),
      peakCurrent: Math.max(...currents),
    };
  }, [chartData]);

  return (
    <div className="app">
      <div className="app__glow" aria-hidden="true" />

      <header className="topbar">
        <div className="brand">
          <span className="brand__mark">
            <Activity size={20} strokeWidth={2.4} />
          </span>
          <span className="brand__text">
            <strong>Carboplatin Sensing</strong>
            <span>AI Platform</span>
          </span>
        </div>

        <span className="status-pill">
          <span className="status-pill__dot" />
          AI Platform · v1.2.0
        </span>
      </header>

      <main className="main">
        <section className="hero">
          <span className="hero__eyebrow">
            <Sparkles size={14} />
            Physics-informed electrochemical analysis
          </span>
          <h1>
            Estimate carboplatin concentration from a single
            <span className="hero__accent"> voltammogram</span>
          </h1>
          <p>
            Upload your potential and current readings. The selected architecture denoises the
            signal, reconstructs the electrochemical response and returns a concentration estimate
            in seconds.
          </p>
        </section>

        <div className="workspace">
          <section className="card upload-card" aria-labelledby="upload-title">
            <div className="card__head">
              <div>
                <h2 id="upload-title">Voltammetry data</h2>
                <p>CSV or TXT file with potential and current columns.</p>
              </div>
              <span className="step-badge">Step 1</span>
            </div>

            <div className="model-selector">
              <div className="model-selector__head">
                <span className="model-selector__label" id="model-architecture-label">
                  Model architecture
                </span>
                <span className="preview-badge">Model routing</span>
              </div>
              <div className="model-options" role="radiogroup" aria-labelledby="model-architecture-label">
                {MODEL_OPTIONS.map((option) => {
                  const Icon = option.icon;
                  const isActive = option.id === architecture;

                  return (
                    <label
                      key={option.id}
                      className={`model-option${isActive ? ' model-option--active' : ''}`}
                    >
                      <input
                        className="visually-hidden"
                        type="radio"
                        name="model-architecture"
                        value={option.id}
                        checked={isActive}
                        onChange={handleArchitectureChange}
                        disabled={loading}
                      />
                      <span className="model-option__icon">
                        <Icon size={16} />
                      </span>
                      <span className="model-option__copy">
                        <strong>{option.name}</strong>
                        <span>{option.tagline}</span>
                      </span>
                      <span className="model-option__check" aria-hidden="true">
                        {isActive && <Check size={13} />}
                      </span>
                    </label>
                  );
                })}
              </div>
            </div>

            <div
              className={`dropzone${isDragging ? ' dropzone--active' : ''}${file ? ' dropzone--filled' : ''}`}
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              <input
                ref={inputRef}
                id="csvFile"
                className="visually-hidden"
                type="file"
                accept=".csv,.txt"
                onChange={handleFileChange}
              />

              {file ? (
                <div className="file-chip">
                  <span className="file-chip__icon">
                    <FileSpreadsheet size={22} />
                  </span>
                  <span className="file-chip__meta">
                    <strong>{file.name}</strong>
                    <span>{formatFileSize(file.size)} · Ready to analyze</span>
                  </span>
                  <button
                    type="button"
                    className="icon-button"
                    onClick={() => inputRef.current?.click()}
                    aria-label="Choose a different file"
                  >
                    <Upload size={16} />
                  </button>
                  <button
                    type="button"
                    className="icon-button icon-button--danger"
                    onClick={handleRemoveFile}
                    aria-label="Remove selected file"
                  >
                    <X size={16} />
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  className="dropzone__action"
                  onClick={() => inputRef.current?.click()}
                >
                  <span className="dropzone__icon">
                    <Upload size={24} />
                  </span>
                  <span className="dropzone__title">Drop your CSV here or browse</span>
                  <span className="dropzone__hint">Two columns: potential (V) and current (µA)</span>
                </button>
              )}
            </div>

            <div className="upload-card__footer">
              <div className="file-spec">
                <FlaskConical size={15} />
                <span>Accepted formats: .csv, .txt</span>
              </div>

              <button
                type="button"
                className="primary-button"
                onClick={handleUpload}
                disabled={!file || loading}
              >
                {loading ? (
                  <>
                    <Loader2 size={18} className="spin" />
                    Analyzing signal…
                  </>
                ) : (
                  <>
                    <Sparkles size={18} />
                    Run analysis
                  </>
                )}
              </button>
            </div>

            {error && (
              <div className="alert" role="alert">
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}
          </section>

          <aside className="card info-card" aria-labelledby="pipeline-title">
            <div className="card__head">
              <div>
                <h2 id="pipeline-title">How it works</h2>
                <p>From raw signal to a concentration reading.</p>
              </div>
              <span className="icon-badge">
                <BrainCircuit size={18} />
              </span>
            </div>

            <ol className="pipeline">
              <li>
                <span className="pipeline__index">1</span>
                <div>
                  <strong>Upload the scan</strong>
                  <p>Export your voltammetry trace with potential and current columns.</p>
                </div>
              </li>
              <li>
                <span className="pipeline__index">2</span>
                <div>
                  <strong>Model inference</strong>
                  <p>{selectedModel.description}</p>
                </div>
              </li>
              <li>
                <span className="pipeline__index">3</span>
                <div>
                  <strong>Review the estimate</strong>
                  <p>Check the predicted concentration, model confidence and residual loss.</p>
                </div>
              </li>
            </ol>

            <div className="chip-row" aria-label="Model stack">
              {selectedModel.tags.map((tag) => (
                <span className="chip" key={tag}>
                  {tag}
                </span>
              ))}
            </div>
          </aside>
        </div>

        {results && (
          <section className="results" aria-labelledby="results-title">
            <div className="results__head">
              <div>
                <span className="results__eyebrow">
                  <CheckCircle2 size={15} />
                  Analysis complete
                </span>
                <h2 id="results-title">Prediction results</h2>
              </div>
              <button type="button" className="ghost-button" onClick={handleReset}>
                New analysis
              </button>
            </div>

            <div className="stat-grid">
              <article className="stat-card stat-card--green">
                <div className="stat-card__top">
                  <span className="stat-card__label">Estimated concentration</span>
                  <span className="stat-card__icon">
                    <Target size={18} />
                  </span>
                </div>
                <div className="stat-card__value">
                  {formatMetric(results.predicted_concentration_uM, 2)}
                  <span>µM</span>
                </div>
                <p className="stat-card__hint">Predicted carboplatin level in the sample</p>
              </article>

              <article className="stat-card stat-card--violet">
                <div className="stat-card__top">
                  <span className="stat-card__label">Model accuracy (R²)</span>
                  <span className="stat-card__icon">
                    <Gauge size={18} />
                  </span>
                </div>
                <div className="stat-card__value">{formatMetric(results.confidence_r2, 3)}</div>
                <div className="meter" aria-hidden="true">
                  <span style={{ width: `${confidencePercent}%` }} />
                </div>
                <p className="stat-card__hint">{confidencePercent}% of variance explained</p>
              </article>

              <article className="stat-card stat-card--amber">
                <div className="stat-card__top">
                  <span className="stat-card__label">Physics residual loss</span>
                  <span className="stat-card__icon">
                    <TrendingDown size={18} />
                  </span>
                </div>
                <div className="stat-card__value">
                  {hasPhysicsLoss ? formatMetric(results.physics_residual_loss, 5) : 'N/A'}
                </div>
                <p className="stat-card__hint">
                  {hasPhysicsLoss
                    ? 'Lower is better — physics constraints satisfied'
                    : 'Not applicable to this architecture'}
                </p>
              </article>
            </div>

            <div className="results-grid">
              <section className="card chart-card" aria-labelledby="chart-title">
                <div className="card__head">
                  <div>
                    <h2 id="chart-title">Voltammogram</h2>
                    <p>Current response across the scanned potential window.</p>
                  </div>
                  <span className="legend">
                    <span className="legend__dot" />
                    Measured current
                  </span>
                </div>

                {chartData.length > 0 ? (
                  <div className="chart">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={chartData} margin={{ top: 10, right: 12, bottom: 0, left: 0 }}>
                        <defs>
                          <linearGradient id="currentFill" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.35} />
                            <stop offset="100%" stopColor="#38bdf8" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" vertical={false} />
                        <XAxis
                          dataKey="potential"
                          stroke="#8092ad"
                          tickLine={false}
                          axisLine={false}
                          tick={{ fontSize: 12 }}
                          tickFormatter={(value) => Number(value).toFixed(2)}
                          height={46}
                          label={{
                            value: 'Potential (V)',
                            position: 'insideBottom',
                            offset: -4,
                            fill: '#8092ad',
                            fontSize: 12,
                          }}
                        />
                        <YAxis
                          stroke="#8092ad"
                          tickLine={false}
                          axisLine={false}
                          tick={{ fontSize: 12 }}
                          tickFormatter={(value) => Number(value).toFixed(1)}
                          width={54}
                          label={{
                            value: 'Current (µA)',
                            angle: -90,
                            position: 'insideLeft',
                            fill: '#8092ad',
                            fontSize: 12,
                            style: { textAnchor: 'middle' },
                          }}
                        />
                        <Tooltip
                          content={<ChartTooltip />}
                          cursor={{ stroke: 'rgba(148, 163, 184, 0.35)', strokeDasharray: '4 4' }}
                        />
                        <Area
                          type="monotone"
                          dataKey="current"
                          stroke="#38bdf8"
                          strokeWidth={2.5}
                          fill="url(#currentFill)"
                          dot={false}
                          activeDot={{ r: 5, fill: '#38bdf8', stroke: '#0b1220', strokeWidth: 3 }}
                          animationDuration={700}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="empty-chart">
                    <Activity size={22} />
                    <span>No signal data was returned for this file.</span>
                  </div>
                )}

                <div className="model-meta">
                  <div className="model-meta__item">
                    <span className="model-meta__icon">
                      <Cpu size={16} />
                    </span>
                    <div>
                      <span>Model architecture</span>
                      <strong>{selectedModel.name}</strong>
                    </div>
                  </div>
                  <div className="model-meta__item">
                    <span className="model-meta__icon">
                      <Database size={16} />
                    </span>
                    <div>
                      <span>Preprocessing pipeline</span>
                      <strong>{results.preprocessing_pipeline || 'Denoising + MinMax scaling'}</strong>
                    </div>
                  </div>
                </div>
              </section>

              <aside className="card summary-card">
                <div className="summary-card__icon">
                  <CheckCircle2 size={20} />
                </div>
                <h2>Signal processed successfully</h2>
                <p>The metrics were computed from the uploaded trace and the reconstructed response.</p>

                <dl className="summary-list">
                  <div>
                    <dt>Data points</dt>
                    <dd>{chartData.length}</dd>
                  </div>
                  <div>
                    <dt>Potential window</dt>
                    <dd>
                      {signalSummary
                        ? `${signalSummary.minPotential.toFixed(2)} – ${signalSummary.maxPotential.toFixed(2)} V`
                        : '—'}
                    </dd>
                  </div>
                  <div>
                    <dt>Peak current</dt>
                    <dd>{signalSummary ? `${signalSummary.peakCurrent.toFixed(2)} µA` : '—'}</dd>
                  </div>
                </dl>

                <button type="button" className="ghost-button ghost-button--full" onClick={handleReset}>
                  Analyze another file
                </button>
              </aside>
            </div>
          </section>
        )}

        <footer className="footer">
          <span>Carboplatin Sensing AI</span>
          <span className="footer__dot" />
          <span>Physics-informed neural network for electrochemical sensing</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
