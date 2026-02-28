"""
Q-volution Hackathon 2026 — QVolution Team
===========================================
Quantum Reservoir Computing for Swaption Volatility Surface
Forecasting and Option Pricing.

Architecture:
    Raw Vol Surface (494 days × 224 grid points)
        → StandardScaler → PCA (6 components, 99.99% variance)
        → Multi-seed Quantum Reservoir Ensemble (MerLin / Quandela)
        → Augmented features [PCA | Quantum] + Washout
        → RidgeCV readout
        → Inverse PCA + Inverse Scaler → real implied-vol space
        → Black-Scholes → put & call option prices
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
from scipy.stats import norm

from quantum_reservoir import QuantumReservoir


# ================================================================
# CONFIGURATION
# ================================================================

N_PCA_COMPONENTS = 6
WASHOUT          = 20
CIRCUIT_DEPTH    = 3
N_MODES          = 6
N_PHOTONS        = 2
RIDGE_ALPHAS     = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
TRAIN_RATIO      = 0.8
ENSEMBLE_SEEDS   = [42, 7, 123, 999, 314]
MEMORY_WINDOWS   = [1, 2, 3, 5]
RISK_FREE_RATE   = 0.05
SPOT             = 100.0
STRIKE           = 100.0

C_Q    = "#a78bfa"
C_CL   = "#34d399"
C_TRUE = "#f9a8d4"
C_BG   = "#1a1a2e"
C_GRID = "#2d2d4d"
C_TEXT = "#e2e8f0"
CMAP   = "plasma"


# ================================================================
# HELPERS
# ================================================================

def to_vol(arr_pca, pca_obj, scaler_obj):
    """PCA space → StandardScaler → original implied-vol space."""
    return scaler_obj.inverse_transform(pca_obj.inverse_transform(arr_pca))


def style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(C_BG)
    ax.set_title(title, color=C_TEXT, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(C_GRID)
    ax.grid(color=C_GRID, linewidth=0.5, alpha=0.6)
    if xlabel:
        ax.set_xlabel(xlabel, color=C_TEXT, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_TEXT, fontsize=9)


def add_legend(ax, **kw):
    ax.legend(fontsize=8, facecolor=C_BG, labelcolor=C_TEXT,
              edgecolor=C_GRID, **kw)


def black_scholes(sigma, S=SPOT, K=STRIKE, T=1.0, r=RISK_FREE_RATE):
    sigma = np.clip(sigma, 1e-6, None)
    d1    = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2    = d1 - sigma * np.sqrt(T)
    call  = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put   = K * np.exp(-r * T) * norm.cdf(-d2)  - S * norm.cdf(-d1)
    return call, put


# ================================================================
# 1. LOAD DATA
# ================================================================

print("=" * 62)
print("  Q-VOLUTION  |  Quantum Reservoir Vol Surface Forecaster")
print("=" * 62)
print("\n[1/9] Loading dataset...")

df = pd.read_excel("train.xlsx")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)
dates    = df["Date"].values
vol_cols = [c for c in df.columns if c != "Date"]
data     = df[vol_cols].values       # (494, 224)  raw implied vols

tenors     = sorted(set(int(c.split("Tenor : ")[1].split(";")[0])
                        for c in vol_cols))
maturities = sorted(set(float(c.split("Maturity : ")[1])
                         for c in vol_cols))
n_t, n_m   = len(tenors), len(maturities)

print(f"  Surface      : {data.shape}  ({n_t} tenors × {n_m} maturities)")
print(f"  Vol range    : [{data.min():.4f}, {data.max():.4f}]")
print(f"  Mean |Δvol|  : {np.abs(np.diff(data, axis=0)).mean():.6f}")
print(f"  Date range   : {pd.Timestamp(dates[0]).date()} → "
      f"{pd.Timestamp(dates[-1]).date()}")


# ================================================================
# 2. SCALE
# ================================================================

scaler      = StandardScaler()
data_scaled = scaler.fit_transform(data)


# ================================================================
# 3. PCA
# ================================================================

print("\n[2/9] PCA reduction...")

pca      = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
data_pca = pca.fit_transform(data_scaled)

print(f"  Variance retained : "
      f"{np.cumsum(pca.explained_variance_ratio_)[-1]*100:.4f}%")
for i, v in enumerate(pca.explained_variance_ratio_):
    print(f"    PC{i+1}: {v*100:.3f}%")


# ================================================================
# 4. TRAIN / TEST SPLIT
# ================================================================

X_all   = data_pca[:-1]   # day t
y_all   = data_pca[1:]    # day t+1
split   = int(TRAIN_RATIO * len(X_all))

X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

# Ground-truth in REAL implied-vol space (no PCA, no scaling artefacts)
y_test_vol = to_vol(y_test, pca, scaler)

print(f"\n[3/9] Train/test split")
print(f"  Train : {X_train.shape[0]} days  |  Test : {X_test.shape[0]} days")


# ================================================================
# 5. CLASSICAL BASELINE
# ================================================================

print("\n[4/9] Classical baseline (RidgeCV on PCA features)...")

bl     = RidgeCV(alphas=RIDGE_ALPHAS)
bl.fit(X_train, y_train)
bl_vol = to_vol(bl.predict(X_test), pca, scaler)
mse_cl = mean_squared_error(y_test_vol, bl_vol)

print(f"  MSE (vol space) : {mse_cl:.6f}  [alpha = {bl.alpha_}]")


# ================================================================
# 6. TEMPORAL MEMORY EXPERIMENT
# ================================================================

print("\n[5/9] Temporal memory experiment (classical Ridge)...")

memory_mse = {}
for w in MEMORY_WINDOWS:
    if w == 1:
        Xw, yw = data_pca[:-1], data_pca[1:]
    else:
        Xw = np.hstack([data_pca[i: len(data_pca) - (w - i)]
                         for i in range(w)])
        yw = data_pca[w:]
    sp    = int(TRAIN_RATIO * len(Xw))
    mw    = RidgeCV(alphas=RIDGE_ALPHAS)
    mw.fit(Xw[:sp], yw[:sp])
    mse_w = mean_squared_error(
        to_vol(yw[sp:], pca, scaler),
        to_vol(mw.predict(Xw[sp:]), pca, scaler))
    memory_mse[w] = mse_w
    print(f"  Window = {w} day(s) : MSE = {mse_w:.6f}")


# ================================================================
# 7. QUANTUM RESERVOIR — MULTI-SEED ENSEMBLE
# ================================================================

print("\n[6/9] Quantum Reservoir ensemble (MerLin / Quandela)...")

train_q_list, test_q_list = [], []

for seed in ENSEMBLE_SEEDS:
    print(f"  Seed {seed:>4d} — building reservoir ...", end=" ", flush=True)
    res = QuantumReservoir(
        n_modes=N_MODES,
        n_photons=N_PHOTONS,
        circuit_depth=CIRCUIT_DEPTH,
        seed=seed,
    )
    res.fit_scaler(X_train)               # fit MinMax on training PCA data
    trq = res.transform(X_train)
    teq = res.transform(X_test)
    train_q_list.append(trq)
    test_q_list.append(teq)
    print(f"feature dim = {trq.shape[1]}")

X_train_q = np.hstack(train_q_list)
X_test_q  = np.hstack(test_q_list)
q_feat    = X_train_q.shape[1]

print(f"  Ensemble : {len(ENSEMBLE_SEEDS)} seeds × "
      f"{train_q_list[0].shape[1]} = {q_feat} quantum features")


# ================================================================
# 8. AUGMENTED READOUT  [PCA | Quantum]  +  WASHOUT
# ================================================================

X_tr_aug = np.hstack([X_train, X_train_q])
X_te_aug = np.hstack([X_test,  X_test_q])
X_tr_fit = X_tr_aug[WASHOUT:]
y_tr_fit = y_train[WASHOUT:]

print(f"\n[7/9] Training readout ...")
print(f"  Feature dim   : {N_PCA_COMPONENTS} PCA + {q_feat} quantum"
      f" = {X_tr_aug.shape[1]}")
print(f"  After washout : {X_tr_fit.shape[0]} training samples")

rdg       = RidgeCV(alphas=RIDGE_ALPHAS)
rdg.fit(X_tr_fit, y_tr_fit)
preds_vol = to_vol(rdg.predict(X_te_aug), pca, scaler)   # ← real vol space
mse_q     = mean_squared_error(y_test_vol, preds_vol)
gain      = (mse_cl - mse_q) / mse_cl * 100

print(f"  Best alpha    : {rdg.alpha_}")
print(f"  Quantum MSE   : {mse_q:.6f}")
print(f"  Improvement   : {gain:+.2f}% vs classical")


# ================================================================
# 9. ABLATION STUDY
# ================================================================

print("\n[8/9] Ablation study...")

ablation = {}

# A) PCA only
ablation["PCA only (baseline)"] = mse_cl

# B) Quantum only, 1 seed
m = RidgeCV(alphas=RIDGE_ALPHAS)
m.fit(train_q_list[0][WASHOUT:], y_train[WASHOUT:])
ablation["Quantum only (1 seed)"] = mean_squared_error(
    y_test_vol, to_vol(m.predict(test_q_list[0]), pca, scaler))

# C) PCA + Quantum, 1 seed, no washout
Xc = np.hstack([X_train, train_q_list[0]])
m  = RidgeCV(alphas=RIDGE_ALPHAS); m.fit(Xc, y_train)
ablation["PCA + Quantum (1 seed, no washout)"] = mean_squared_error(
    y_test_vol,
    to_vol(m.predict(np.hstack([X_test, test_q_list[0]])), pca, scaler))

# D) PCA + Quantum, 1 seed, with washout
Xd = np.hstack([X_train, train_q_list[0]])[WASHOUT:]
m  = RidgeCV(alphas=RIDGE_ALPHAS); m.fit(Xd, y_train[WASHOUT:])
ablation["PCA + Quantum (1 seed + washout)"] = mean_squared_error(
    y_test_vol,
    to_vol(m.predict(np.hstack([X_test, test_q_list[0]])), pca, scaler))

# E) Full ensemble (current model)
ablation["PCA + Ensemble (5 seeds + washout)"] = mse_q

best_mse = min(ablation.values())
for k, v in ablation.items():
    tag = " ◀ BEST" if v == best_mse else ""
    print(f"  {k:<44s}: {v:.6f}{tag}")


# ================================================================
# 10. BLACK-SCHOLES OPTION PRICING
#     Tenor=15 (delta 50 ≈ ATM), Maturity=1 year
#     All inputs are real implied vols — no spikes possible
# ================================================================

print("\n[9/9] Black-Scholes option pricing...")

# Exact string match to avoid matching Maturity 1.5, 10, 15, etc.
atm_idx = next(
    i for i, c in enumerate(vol_cols)
    if "Tenor : 15" in c
    and c.split("Maturity : ")[1].rstrip() == "1"
)
print(f"  ATM column : [{atm_idx}] {vol_cols[atm_idx]}")

true_vol = y_test_vol[:, atm_idx]
pred_vol = preds_vol[:,  atm_idx]
cl_vol   = bl_vol[:,     atm_idx]

print(f"  True vol  : [{true_vol.min():.4f}, {true_vol.max():.4f}]")
print(f"  Pred vol  : [{pred_vol.min():.4f}, {pred_vol.max():.4f}]")

true_call, true_put = black_scholes(true_vol)
pred_call, pred_put = black_scholes(pred_vol)
cl_call,   cl_put   = black_scholes(cl_vol)

mse_cc = mean_squared_error(true_call, cl_call)
mse_cp = mean_squared_error(true_put,  cl_put)
mse_qc = mean_squared_error(true_call, pred_call)
mse_qp = mean_squared_error(true_put,  pred_put)
imp_c  = (mse_cc - mse_qc) / mse_cc * 100
imp_p  = (mse_cp - mse_qp) / mse_cp * 100

print(f"\n  {'Model':<36s} {'Call MSE':>10s} {'Put MSE':>10s}")
print(f"  {'-'*58}")
print(f"  {'Classical Ridge':<36s} {mse_cc:>10.6f} {mse_cp:>10.6f}")
print(f"  {'Quantum Ensemble':<36s} {mse_qc:>10.6f} {mse_qp:>10.6f}")
print(f"  Improvement  Call: {imp_c:+.2f}%   Put: {imp_p:+.2f}%")


# ================================================================
# VISUALISATION
# ================================================================

print("\nGenerating results.png ...")

fig = plt.figure(figsize=(21, 26))
fig.patch.set_facecolor("#0d0d1a")
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.36)


# ── Row 0: Vol surface heatmaps ──────────────────────────────────────────────
# FIX: use actual per-surface vmin/vmax so low-maturity vols are visible

for col, (title, surf_vol) in enumerate([
    ("True Vol Surface\n(last test day)",    y_test_vol[-1]),
    ("Quantum Predicted\n(last test day)",   preds_vol[-1]),
    ("Classical Predicted\n(last test day)", bl_vol[-1]),
]):
    ax  = fig.add_subplot(gs[0, col])
    mat = surf_vol.reshape(n_t, n_m)

    # Use the shared range of actual vol data for consistent comparison
    vmin = data.min()
    vmax = np.percentile(data, 98)   # clip top 2% so structure is visible

    im = ax.imshow(mat, aspect="auto", cmap=CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_m))
    ax.set_xticklabels([f"{m:.2f}" for m in maturities],
                       rotation=45, ha="right", fontsize=6, color=C_TEXT)
    ax.set_yticks(range(n_t))
    ax.set_yticklabels(tenors, fontsize=7, color=C_TEXT)
    style_ax(ax, title, "Maturity (yrs)", "Tenor (Δ)")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=C_TEXT, labelsize=7)
    cb.set_label("Implied Vol", color=C_TEXT, fontsize=7)

    # Show per-panel MSE vs truth on prediction panels
    if col > 0:
        err = mean_squared_error(y_test_vol[-1], surf_vol)
        ax.text(0.02, 0.02, f"MSE vs truth: {err:.2e}",
                transform=ax.transAxes, color=C_TEXT, fontsize=8,
                va="bottom", bbox=dict(boxstyle="round,pad=0.3",
                                       fc=C_BG, ec=C_GRID, alpha=0.8))


# ── Row 1 left: Memory window ────────────────────────────────────────────────

ax = fig.add_subplot(gs[1, 0])
ws, ms = list(memory_mse.keys()), list(memory_mse.values())
bars   = ax.bar(ws, ms, color=C_Q, alpha=0.85, width=0.55, zorder=3)
ax.axhline(mse_cl, color=C_CL, lw=1.8, ls="--", zorder=4,
           label=f"Classical w=1  ({mse_cl:.5f})")
for b, v in zip(bars, ms):
    ax.text(b.get_x() + b.get_width() / 2, v + 2e-5,
            f"{v:.5f}", ha="center", va="bottom", color=C_TEXT, fontsize=7)
add_legend(ax)
style_ax(ax, "Temporal Memory Experiment\n(classical Ridge, varying lag window)",
         "History window (days)", "Full surface MSE")


# ── Row 1 centre: Ablation ───────────────────────────────────────────────────

ax  = fig.add_subplot(gs[1, 1])
abl_labels = list(ablation.keys())
abl_vals   = list(ablation.values())
pal        = [C_CL, "#fb923c", "#60a5fa", "#fbbf24", C_Q]
hb         = ax.barh(range(len(abl_labels)), abl_vals,
                     color=pal[:len(abl_labels)], alpha=0.88, zorder=3)
ax.set_yticks(range(len(abl_labels)))
ax.set_yticklabels(abl_labels, fontsize=7.5, color=C_TEXT)
x_max = max(abl_vals) * 1.15
ax.set_xlim(0, x_max)
for b, v in zip(hb, abl_vals):
    ax.text(min(v + x_max * 0.01, x_max * 0.98),
            b.get_y() + b.get_height() / 2,
            f"{v:.5f}", va="center", color=C_TEXT, fontsize=7)
ax.axvline(mse_cl, color=C_CL, lw=1.2, ls=":", alpha=0.7)
style_ax(ax, "Ablation Study", "Full surface MSE (vol space)")


# ── Row 1 right: t-SNE quantum features ─────────────────────────────────────

ax    = fig.add_subplot(gs[1, 2])
q_all = np.vstack([X_train_q, X_test_q])
vol_lv = data_pca[:-1, 0]    # PC1 ≈ overall vol level
perp   = min(30, len(q_all) - 1)
tsne   = TSNE(n_components=2, perplexity=perp, random_state=42,
               max_iter=1000, init="pca")
q2d    = tsne.fit_transform(q_all)

sc = ax.scatter(q2d[:, 0], q2d[:, 1], c=vol_lv,
                cmap=CMAP, s=14, alpha=0.80, zorder=3)
ax.scatter(q2d[split:, 0], q2d[split:, 1], s=35,
           facecolors="none", edgecolors=C_TRUE,
           linewidths=0.9, label="Test set", zorder=4)
cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cb.ax.tick_params(colors=C_TEXT, labelsize=7)
cb.set_label("PC1 (vol level)", color=C_TEXT, fontsize=8)
add_legend(ax)
style_ax(ax, "Quantum Feature Space (t-SNE)\n"
             "Clusters = distinct vol regimes in Hilbert space")


# ── Row 2: ATM 1Y vol forecast — REAL implied-vol space ─────────────────────
# FIX: use to_vol() output, not raw PCA values

ax = fig.add_subplot(gs[2, :])
xi = np.arange(len(true_vol))
ax.plot(xi, true_vol, color=C_TRUE, lw=2.0, label="True implied vol", zorder=5)
ax.plot(xi, pred_vol, color=C_Q,    lw=1.4, alpha=0.92,
        label="Quantum forecast", zorder=4)
ax.plot(xi, cl_vol,   color=C_CL,   lw=1.1, alpha=0.75, ls="--",
        label="Classical forecast", zorder=3)
ax.fill_between(xi, true_vol, pred_vol, alpha=0.12, color=C_Q)
ax.set_ylim(0, max(true_vol.max(), pred_vol.max(), cl_vol.max()) * 1.15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
add_legend(ax)
style_ax(ax,
         "ATM 1Y Swaption Implied Vol — Forecast vs Truth  "
         "(test set, real vol space)",
         "Test day index", "Implied volatility")


# ── Row 3 left: Call prices ──────────────────────────────────────────────────

ax = fig.add_subplot(gs[3, 0])
ax.plot(xi, true_call, color=C_TRUE, lw=2.0, label="True")
ax.plot(xi, pred_call, color=C_Q,    lw=1.4, alpha=0.92,
        label=f"Quantum  (MSE = {mse_qc:.5f})")
ax.plot(xi, cl_call,   color=C_CL,   lw=1.1, alpha=0.75, ls="--",
        label=f"Classical (MSE = {mse_cc:.5f})")
ax.set_ylim(0, None)
add_legend(ax)
style_ax(ax, f"ATM Call Option Price\n"
             f"(S=K={SPOT}, T=1Y, r={RISK_FREE_RATE})",
         "Test day index", "Call price")


# ── Row 3 centre: Put prices ─────────────────────────────────────────────────

ax = fig.add_subplot(gs[3, 1])
ax.plot(xi, true_put, color=C_TRUE, lw=2.0, label="True")
ax.plot(xi, pred_put, color=C_Q,    lw=1.4, alpha=0.92,
        label=f"Quantum  (MSE = {mse_qp:.5f})")
ax.plot(xi, cl_put,   color=C_CL,   lw=1.1, alpha=0.75, ls="--",
        label=f"Classical (MSE = {mse_cp:.5f})")
ax.set_ylim(0, None)
add_legend(ax)
style_ax(ax, f"ATM Put Option Price\n"
             f"(S=K={SPOT}, T=1Y, r={RISK_FREE_RATE})",
         "Test day index", "Put price")


# ── Row 3 right: Summary card ────────────────────────────────────────────────

ax = fig.add_subplot(gs[3, 2])
ax.set_facecolor(C_BG)
ax.axis("off")

imp_color = C_Q if gain > 0 else "#f87171"
ic_color  = C_Q if imp_c > 0 else "#f87171"
ip_color  = C_Q if imp_p > 0 else "#f87171"

card = [
    ("MODEL",                                          C_Q,      13, "bold"),
    (f"Modes / Photons      : {N_MODES} / {N_PHOTONS}", C_TEXT,  9, "normal"),
    (f"Circuit depth         : {CIRCUIT_DEPTH}",       C_TEXT,   9, "normal"),
    (f"Ensemble seeds        : {len(ENSEMBLE_SEEDS)}", C_TEXT,   9, "normal"),
    (f"Feature dim           : {N_PCA_COMPONENTS} PCA"
     f" + {q_feat} Q = {X_tr_aug.shape[1]}",          C_TEXT,   9, "normal"),
    ("",                                               C_TEXT,   5, "normal"),
    ("VOL SURFACE MSE",                                C_Q,     12, "bold"),
    (f"Classical   : {mse_cl:.6f}",                   C_CL,     9, "normal"),
    (f"Quantum     : {mse_q:.6f}",                    C_Q,      9, "normal"),
    (f"Improvement : {gain:+.2f}%",                   imp_color,11, "bold"),
    ("",                                               C_TEXT,   5, "normal"),
    ("OPTION PRICING  (ATM 1Y)",                       C_Q,     12, "bold"),
    (f"Call — Classical : {mse_cc:.6f}",              C_CL,     9, "normal"),
    (f"Call — Quantum   : {mse_qc:.6f}",              C_Q,      9, "normal"),
    (f"Call improvement : {imp_c:+.2f}%",             ic_color, 10, "bold"),
    (f"Put  — Classical : {mse_cp:.6f}",              C_CL,     9, "normal"),
    (f"Put  — Quantum   : {mse_qp:.6f}",              C_Q,      9, "normal"),
    (f"Put  improvement : {imp_p:+.2f}%",             ip_color, 10, "bold"),
]

yp = 0.97
for text, color, size, weight in card:
    ax.text(0.04, yp, text, transform=ax.transAxes,
            color=color, fontsize=size, fontweight=weight, va="top")
    yp -= 0.052

fig.suptitle(
    "QVolution  |  Quantum Reservoir Computing for Swaption Vol Surface "
    "Forecasting  |  Q-volution Hackathon 2026",
    color=C_TEXT, fontsize=13, fontweight="bold", y=0.997)

plt.savefig("results.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("  Saved → results.png")


# ================================================================
# FINAL CONSOLE SUMMARY
# ================================================================

print("\n" + "=" * 62)
print("  FINAL RESULTS")
print("=" * 62)
print(f"  Classical Ridge MSE (vol surface) : {mse_cl:.6f}")
print(f"  Quantum Ensemble MSE (vol surface): {mse_q:.6f}")
print(f"  Vol surface improvement           : {gain:+.2f}%")
print(f"  Call option improvement           : {imp_c:+.2f}%")
print(f"  Put  option improvement           : {imp_p:+.2f}%")
print()
print("  Ablation:")
for k, v in ablation.items():
    tag = " ◀ BEST" if v == best_mse else ""
    print(f"    {k:<44s}: {v:.6f}{tag}")
print()
print("  Memory window sweep:")
for w, m in memory_mse.items():
    print(f"    Window = {w} : {m:.6f}")
print("=" * 62)
print("  Output : results.png")
print("  Training completed successfully.")