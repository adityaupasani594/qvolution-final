

A quantum-powered volatility forecasting application combining a FastAPI backend with a React/Vite frontend. It uses a quantum reservoir computing model (via MerLin/Quandela) to predict implied volatility curves and option pricing via Black-Scholes.

---

## Prerequisites

- Python 3.9+
- Node.js 18+ and npm

---

## Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/adityaupasani594/qvolution-final
cd qvolution-final-main
```

### 2. Backend

**Install Python dependencies:**

```bash
pip install -r requirements.txt
```

**Start the API server:**

```bash
uvicorn backend_api:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`.  
Health check: `http://localhost:8000/health`

> **Note:** To run on a real Quandela QPU (Belenos), set your API token as an environment variable before starting the server:
> ```bash
> # Windows
> set QUANDELA_TOKEN=your-quandela-cloud-api-token
>
> # macOS/Linux
> export QUANDELA_TOKEN=your-quandela-cloud-api-token
> ```
> If no token is set, the local MerLin statevector simulator is used instead.

---

### 3. Frontend

```bash
cd qvolution-frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.

---

## Project Structure

```
├── backend_api.py              # FastAPI server — /health and /predict endpoints
├── quantum_reservoir.py        # Quantum reservoir computing model
├── features.py                 # Feature engineering utilities
├── DeepLearning_Counterpart.py # Classical deep learning baseline
├── validate.py                 # Model validation scripts
├── requirements.txt            # Python dependencies
└── qvolution-frontend/         # React + Vite frontend
    ├── src/
    │   ├── App.jsx
    │   ├── InputPage.jsx
    │   └── Dashboard.jsx
    ├── index.html
    └── package.json
```

---

## Technical Details

### Architecture Overview

```
Raw Volatility Surface (494 days × 224 grid points)
    → StandardScaler
    → PCA (6 components, retaining 99.99% of variance)
    → Multi-seed Quantum Reservoir Ensemble
    → Augmented feature matrix [PCA | Quantum] + washout
    → RidgeCV readout layer
    → Inverse PCA + Inverse Scaler → implied-volatility space
    → Black-Scholes → call & put option prices
```

### Quantum Reservoir Computing (QORC)

The core model is a **Quantum Optical Reservoir Computer** built on [MerLin](https://merlinquantum.com/) (Quandela):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_modes` | 6 | Number of optical modes in the photonic circuit |
| `n_photons` | 2 | Number of photons injected (Fock state `\|1,1,0,0,0,0⟩`) |
| `circuit_depth` | 3 | Depth of entangling + superposition layers |
| `ensemble_seeds` | [42, 7, 123, 999, 314] | 5 independent random reservoirs ensemble-averaged |
| `washout` | 20 | Initial time steps discarded to allow reservoir transience |
| `output_dim` | C(n_modes + n_photons − 1, n_photons) = **21** | Fock-space probability feature dimension per reservoir |

**Encoding:** Each PCA component is MinMax-scaled to `[0, 1]`, then encoded as a phase shift $\theta = x \cdot \pi$ into the photonic circuit. The reservoir weights (interferometer) are **fixed and untrained** — only the RidgeCV readout is optimised.

**Measurement:** Fock-space photon-number probability distribution (21-dimensional vector per sample).

### PCA Compression

- Input: 224-dimensional implied volatility grid per day
- Compressed to **6 PCA components** capturing 99.99% of variance
- Reduces quantum circuit calls from 224 to 6 per time step

### Readout & Regularisation

- **RidgeCV** with cross-validated alpha selection from `[0.001, 0.01, 0.1, 1, 10, 100, 1000]`
- Trained on the concatenated `[PCA features | quantum reservoir features]` matrix
- Memory windows `[1, 2, 3, 5]` days of lag features are appended for temporal context

### Black-Scholes Pricing

Predicted volatility $\hat{\sigma}$ is fed into the **Black-Scholes** formula to produce call and put prices:

$$d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

$$C = S\,\Phi(d_1) - K e^{-rT}\Phi(d_2), \quad P = K e^{-rT}\Phi(-d_2) - S\,\Phi(-d_1)$$

Default parameters: $S = K = 100$, $r = 0.05$, $T = 1.0$ (1 year).

### Classical Baseline

`DeepLearning_Counterpart.py` implements a **Classical Ensemble Reservoir** that mirrors the quantum architecture exactly — same Fock-space output dimension, same fixed random weights (frozen), MinMax scaling + tanh activation — allowing a direct apples-to-apples performance comparison.

### Execution Backends

| Mode | How to activate | Notes |
|------|----------------|-------|
| Local simulator | Default (no token set) | MerLin statevector simulator; exact probabilities |
| Quandela Belenos QPU | Set `QUANDELA_TOKEN` env var | 10 000 shots per circuit; sampling noise included |

---

## API Reference

### `POST /predict`

Accepts 224 historical volatility values and returns a smoothed predicted curve, a 30-day price forecast, and mean call/put prices.

**Request body:**
```json
{
  "values": [/* 224 floats */]
}
```

**Response:**
```json
{
  "predicted_curve": [...],
  "price_forecast": [{ "day": 1, "call": 10.5, "put": 9.8 }, ...],
  "put_call_prices": [{ "name": "Call", "price": 10.5 }, { "name": "Put", "price": 9.8 }]
}
```

---

## Running Both Services Together

Open two terminals and run the backend and frontend commands simultaneously:

| Terminal | Command |
|----------|---------|
| Terminal 1 | `uvicorn backend_api:app --reload --port 8000` |
| Terminal 2 | `cd qvolution-frontend && npm run dev` |

Then open `http://localhost:5173` in your browser.


