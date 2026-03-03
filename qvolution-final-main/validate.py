"""
validate.py  --  Quantum Reservoir validation on Test.csv
=========================================================
Loads the model trained by features.py (model_artifacts.pkl) and applies
it to Test.csv, printing all validation metrics to the terminal.

Workflow:
    1. Run features.py on train.xlsx  -->  produces model_artifacts.pkl
    2. Run this script on Test.csv    -->  loads artifacts, runs inference

Usage:
    python validate.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import joblib
from math import comb
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from quantum_reservoir import QuantumReservoir

# ================================================================
# CONFIG
# ================================================================

CSV_PATH       = "Test.csv"
ARTIFACTS_PATH = "model_artifacts.pkl"

# ================================================================
# HELPERS
# ================================================================

def to_vol(arr_pca, pca_obj, scaler_obj):
    """PCA space --> StandardScaler --> original implied-vol space."""
    return scaler_obj.inverse_transform(pca_obj.inverse_transform(arr_pca))


def black_scholes(sigma, S, K, T=1.0, r=0.05):
    sigma = np.clip(sigma, 1e-6, None)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call, put


def section(title):
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def subsection(title):
    print(f"\n  -- {title}")


# ================================================================
# 1. LOAD ARTIFACTS
# ================================================================

section("Q-VOLUTION  |  Quantum Reservoir Validation  |  Test.csv")

if not os.path.exists(ARTIFACTS_PATH):
    print(f"\n[ERROR] {ARTIFACTS_PATH} not found.")
    print("  Run features.py on train.xlsx first to generate the trained model.")
    sys.exit(1)

print(f"\n[1/6] Loading trained model from {ARTIFACTS_PATH} ...")

art         = joblib.load(ARTIFACTS_PATH)
scaler      = art["scaler"]
pca         = art["pca"]
rdg         = art["rdg"]
res_scalers = art["reservoir_scalers"]
vol_cols    = art["vol_cols"]
cfg         = art["config"]

N_MODES          = cfg["N_MODES"]
N_PHOTONS        = cfg["N_PHOTONS"]
CIRCUIT_DEPTH    = cfg["CIRCUIT_DEPTH"]
N_PCA_COMPONENTS = cfg["N_PCA_COMPONENTS"]
ENSEMBLE_SEEDS   = cfg["ENSEMBLE_SEEDS"]
WASHOUT          = cfg["WASHOUT"]
RISK_FREE_RATE   = cfg["RISK_FREE_RATE"]
SPOT             = cfg["SPOT"]
STRIKE           = cfg["STRIKE"]

rdg_n_feat = rdg.coef_.shape[1] if rdg.coef_.ndim > 1 else len(rdg.coef_)
print(f"  Model config     : {N_MODES} modes / {N_PHOTONS} photons / depth {CIRCUIT_DEPTH}")
print(f"  PCA components   : {N_PCA_COMPONENTS}")
print(f"  Ensemble seeds   : {ENSEMBLE_SEEDS}")
print(f"  Fock dim / seed  : {comb(N_MODES + N_PHOTONS - 1, N_PHOTONS)}")
print(f"  Ridge alpha      : {rdg.alpha_:.4e}")
print(f"  Readout features : {rdg_n_feat}")

# ================================================================
# 2. LOAD TEST DATA
# ================================================================

print(f"\n[2/6] Loading {CSV_PATH} ...")

if not os.path.exists(CSV_PATH):
    print(f"[ERROR] {CSV_PATH} not found.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

dates     = df["Date"].values
test_cols = [c for c in df.columns if c != "Date"]

missing = [c for c in vol_cols if c not in test_cols]
extra   = [c for c in test_cols if c not in vol_cols]
if missing:
    print(f"  [WARN] {len(missing)} column(s) from training missing in Test.csv -- filling with 0")
if extra:
    print(f"  [WARN] {len(extra)} extra column(s) in Test.csv not in training set -- dropping")

data_df = df.reindex(columns=["Date"] + vol_cols).drop(columns=["Date"])
data_df = data_df.fillna(0)
data    = data_df.values.astype(float)

n_rows, n_features = data.shape

tenors     = sorted(set(int(c.split("Tenor : ")[1].split(";")[0]) for c in vol_cols))
maturities = sorted(set(float(c.split("Maturity : ")[1])           for c in vol_cols))
n_t, n_m   = len(tenors), len(maturities)

print(f"  Rows / Features  : {n_rows} days x {n_features} grid points")
print(f"  Grid             : {n_t} tenors x {n_m} maturities")
print(f"  Vol range        : [{data.min():.6f}, {data.max():.6f}]")
print(f"  Mean |Dvol|      : {np.abs(np.diff(data, axis=0)).mean():.8f}")
print(f"  Date range       : {pd.Timestamp(dates[0]).date()} -> "
      f"{pd.Timestamp(dates[-1]).date()}")

if n_rows < 2:
    print("[ERROR] Need at least 2 rows (1 input day + 1 target day).")
    sys.exit(1)

# ================================================================
# 3. PREPROCESS  (training scaler + PCA, no refit)
# ================================================================

print(f"\n[3/6] Preprocessing with training scaler & PCA (no refit) ...")

data_scaled = scaler.transform(data)
data_pca    = pca.transform(data_scaled)   # (n_rows, N_PCA_COMPONENTS)

print(f"  PCA output shape : {data_pca.shape}")

X_test = data_pca[:-1]   # day t
y_test = data_pca[1:]    # day t+1 (PCA space)

y_test_vol = to_vol(y_test, pca, scaler)

n_pairs = len(X_test)
print(f"  (X, y) pairs     : {n_pairs}")
print(f"  Target date(s)   : "
      + ", ".join(str(pd.Timestamp(dates[i+1]).date()) for i in range(n_pairs)))

# ================================================================
# 4. QUANTUM RESERVOIR  (inference only, restored scalers)
# ================================================================

print(f"\n[4/6] Running Quantum Reservoir ensemble inference "
      f"({len(ENSEMBLE_SEEDS)} seeds, local simulator) ...")

os.environ["QUANDELA_USE_QPU"] = "false"

test_q_list = []

for seed, saved_scaler in zip(ENSEMBLE_SEEDS, res_scalers):
    print(f"  Seed {seed:>4d} -- rebuilding reservoir ...", end=" ", flush=True)
    res = QuantumReservoir(
        n_modes=N_MODES,
        n_photons=N_PHOTONS,
        circuit_depth=CIRCUIT_DEPTH,
        seed=seed,
        use_qpu=False,
    )
    # Restore the fitted input scaler from training -- no refit on test data
    res._input_scaler  = saved_scaler
    res._scaler_fitted = True
    teq = res.transform(X_test)
    test_q_list.append(teq)
    print(f"feature dim = {teq.shape[1]}")

X_test_q = np.hstack(test_q_list)
q_feat   = X_test_q.shape[1]
print(f"\n  Ensemble : {len(ENSEMBLE_SEEDS)} seeds x "
      f"{test_q_list[0].shape[1]} = {q_feat} quantum features")

# ================================================================
# 5. PREDICT
# ================================================================

X_te_aug  = np.hstack([X_test, X_test_q])   # [PCA | Quantum]
preds_pca = rdg.predict(X_te_aug)
preds_vol = to_vol(preds_pca, pca, scaler)

print(f"\n[5/6] Predictions generated for {n_pairs} day(s).")

# ================================================================
# 6. METRICS
# ================================================================

print(f"\n[6/6] Computing validation metrics ...")

mse_q  = mean_squared_error(y_test_vol, preds_vol)
rmse_q = np.sqrt(mse_q)
mae_q  = mean_absolute_error(y_test_vol, preds_vol)
r2_q   = r2_score(y_test_vol, preds_vol) if n_pairs > 1 else float("nan")

print()
print(f"  {'Metric':<30s} {'Value':>14s}")
print(f"  {'-' * 46}")
print(f"  {'Ridge alpha (from training)':<30s} {rdg.alpha_:>14.4e}")
print(f"  {'Vol surface MSE':<30s} {mse_q:>14.8f}")
print(f"  {'Vol surface RMSE':<30s} {rmse_q:>14.8f}")
print(f"  {'Vol surface MAE':<30s} {mae_q:>14.8f}")
if not np.isnan(r2_q):
    print(f"  {'Vol surface R2':<30s} {r2_q:>14.6f}")
else:
    print(f"  {'Vol surface R2 (N/A: 1 sample)':<30s} {'N/A':>14s}")

if n_pairs > 1:
    subsection("Per-day vol surface MSE")
    print(f"    {'Day':<6} {'Date':<14} {'MSE':>12} {'MAE':>12}")
    print(f"    {'-' * 46}")
    for i in range(n_pairs):
        d_mse = mean_squared_error(y_test_vol[i], preds_vol[i])
        d_mae = mean_absolute_error(y_test_vol[i], preds_vol[i])
        print(f"    {i+1:<6} {str(pd.Timestamp(dates[i+1]).date()):<14} "
              f"{d_mse:>12.8f} {d_mae:>12.8f}")

subsection("Per-PC metrics (PCA space)")
print(f"    {'PC':<6} {'MSE':>14} {'R2':>10}")
print(f"    {'-' * 32}")
for i in range(N_PCA_COMPONENTS):
    pc_mse = mean_squared_error(y_test[:, i], preds_pca[:, i])
    pc_r2  = r2_score(y_test[:, i], preds_pca[:, i]) if n_pairs > 1 else float("nan")
    r2_str = f"{pc_r2:>10.6f}" if not np.isnan(pc_r2) else "       N/A"
    print(f"    PC{i+1:<5} {pc_mse:>14.8f} {r2_str}")

subsection("Black-Scholes option pricing  (ATM, Tenor=15, Maturity=1Y)")
mse_qc = mse_qp = None
try:
    atm_idx = next(
        i for i, c in enumerate(vol_cols)
        if "Tenor : 15" in c and c.split("Maturity : ")[1].rstrip() == "1"
    )
    atm_col  = vol_cols[atm_idx]
    true_vol = y_test_vol[:, atm_idx]
    pred_vol = preds_vol[:,  atm_idx]
    true_call, true_put = black_scholes(true_vol, SPOT, STRIKE, r=RISK_FREE_RATE)
    pred_call, pred_put = black_scholes(pred_vol, SPOT, STRIKE, r=RISK_FREE_RATE)
    mse_qc = mean_squared_error(true_call, pred_call)
    mse_qp = mean_squared_error(true_put,  pred_put)
    print(f"    ATM column : [{atm_idx}] {atm_col}")
    print(f"    S={SPOT}, K={STRIKE}, r={RISK_FREE_RATE}, T=1Y")
    print()
    print(f"    {'Model':<28s} {'Call MSE':>12s} {'Put MSE':>12s}")
    print(f"    {'-' * 54}")
    print(f"    {'Quantum Ensemble':<28s} {mse_qc:>12.8f} {mse_qp:>12.8f}")
    print()
    for j in range(n_pairs):
        print(f"    Day {j+1}  ({pd.Timestamp(dates[j+1]).date()})")
        print(f"      True  vol : {true_vol[j]:.6f}   Pred vol : {pred_vol[j]:.6f}   "
              f"Error : {abs(true_vol[j]-pred_vol[j]):.8f}")
        print(f"      True call : {true_call[j]:.6f}   Pred call: {pred_call[j]:.6f}   "
              f"Error : {abs(true_call[j]-pred_call[j]):.8f}")
        print(f"      True put  : {true_put[j]:.6f}   Pred put : {pred_put[j]:.6f}   "
              f"Error : {abs(true_put[j]-pred_put[j]):.8f}")
except StopIteration:
    print("    [WARN] ATM column (Tenor=15, Maturity=1) not found -- skipping option pricing.")

subsection("Vol surface snapshot  (all test days)")
for j in range(n_pairs):
    tru = y_test_vol[j]
    pre = preds_vol[j]
    print(f"    Day {j+1}  ({pd.Timestamp(dates[j+1]).date()})")
    print(f"      True -- mean:{tru.mean():.6f}  std:{tru.std():.6f}  "
          f"range:[{tru.min():.4f}, {tru.max():.4f}]")
    print(f"      Pred -- mean:{pre.mean():.6f}  std:{pre.std():.6f}  "
          f"range:[{pre.min():.4f}, {pre.max():.4f}]")
    print(f"      Grid MSE:{mean_squared_error(tru,pre):.8f}   "
          f"MAE:{mean_absolute_error(tru,pre):.8f}")

subsection("Model configuration (loaded from training)")
print(f"    Modes          : {N_MODES}")
print(f"    Photons        : {N_PHOTONS}")
print(f"    Circuit depth  : {CIRCUIT_DEPTH}")
print(f"    PCA components : {N_PCA_COMPONENTS}")
print(f"    Ensemble seeds : {ENSEMBLE_SEEDS}")
print(f"    Washout (train): {WASHOUT}")
print(f"    Ridge alpha    : {rdg.alpha_:.4e}")
print(f"    Feature dim    : {N_PCA_COMPONENTS} PCA + {q_feat} quantum = {X_te_aug.shape[1]}")
print(f"    Backend        : local MerLin statevector simulator")
print(f"    QPU            : disabled (use_qpu=False)")
print(f"    Artifacts from : {ARTIFACTS_PATH}")

# ================================================================
# FINAL SUMMARY
# ================================================================

section("VALIDATION RESULTS SUMMARY")
print(f"  Artifacts        : {ARTIFACTS_PATH}")
print(f"  Test data        : {CSV_PATH}  ({n_rows} days, {n_features} implied vols)")
print(f"  Test day(s)      : "
      + ", ".join(str(pd.Timestamp(dates[i+1]).date()) for i in range(n_pairs)))
print()
print(f"  Vol surface MSE  : {mse_q:.8f}")
print(f"  Vol surface RMSE : {rmse_q:.8f}")
print(f"  Vol surface MAE  : {mae_q:.8f}")
if not np.isnan(r2_q):
    print(f"  Vol surface R2   : {r2_q:.6f}")
else:
    print(f"  Vol surface R2   : N/A (need >1 test sample)")
if mse_qc is not None:
    print()
    print(f"  ATM Call MSE     : {mse_qc:.8f}")
    print(f"  ATM Put  MSE     : {mse_qp:.8f}")
print()
print("  Validation completed successfully.")
print("=" * 64)
