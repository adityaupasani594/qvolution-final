import joblib
from math import comb

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, KFold
from sklearn.preprocessing import MinMaxScaler

# ==============================
# 1. LOAD DATA
# ==============================

data = pd.read_csv("transformed_dataset.csv")

X = data.iloc[:, :6].values
y = data.iloc[:, 6:].values

split_idx = int(0.8 * len(X))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ==============================
# 2. CLASSICAL ENSEMBLE RESERVOIR
# ==============================

class ClassicalReservoir:
    """
    Classical Ensemble Reservoir Computing
    ---------------------------------------
    - Fixed random weight matrix W (frozen — mirrors quantum interferometer)
    - MinMax [0,1] input scaling  →  tanh activation
      (mirrors angle encoding + Fock-space probability measurement)
    - output_dim = C(n_modes + n_photons - 1, n_photons)  →  matches Fock-space dim
    - Input scaler — must call fit_scaler() before transform()
    """

    def __init__(
        self,
        n_modes=6,
        n_photons=2,
        circuit_depth=2,
        seed=42,
    ):
        self.n_modes       = n_modes
        self.n_photons     = n_photons
        self.circuit_depth = circuit_depth

        np.random.seed(seed)

        # Output dim mirrors Fock-space: C(n_modes + n_photons - 1, n_photons)
        self.output_dim = comb(n_modes + n_photons - 1, n_photons)

        # Fixed random weights — frozen, mirrors quantum interferometer
        self.W = np.random.randn(n_modes, self.output_dim)

        # Input scaler — must call fit_scaler() before transform()
        self._input_scaler  = MinMaxScaler(feature_range=(0, 1))
        self._scaler_fitted = False

        print(f"Reservoir feature dim: {self.output_dim}")

    # -----------------------------------------------------------
    # Input Scaling
    # -----------------------------------------------------------

    def fit_scaler(self, X_train: np.ndarray):
        """
        Fit the MinMaxScaler on training data so inputs are normalised
        to [0, 1] before encoding — mirrors QuantumReservoir.fit_scaler().
        Must be called BEFORE transform().
        """
        self._input_scaler.fit(X_train)
        self._scaler_fitted = True
        return self

    # -----------------------------------------------------------
    # Input Encoding
    # -----------------------------------------------------------

    def _encode(self, x: np.ndarray) -> np.ndarray:
        if not self._scaler_fitted:
            raise RuntimeError(
                "Input scaler not fitted. Call fit_scaler(X_train) before transform()."
            )
        # MinMax → [0, 1]; mirrors quantum angle encoding (scale=π absorbed into W)
        x_scaled = self._input_scaler.transform(x.reshape(1, -1)).flatten()
        return x_scaled  # shape (n_modes,)

    # -----------------------------------------------------------
    # Feature Transformation
    # -----------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)

        Returns
        -------
        features : numpy array of shape (n_samples, output_dim)
        """
        features = []

        for idx, sample in enumerate(X):
            if idx % 50 == 0:
                print(f"  Processing sample {idx}/{len(X)}...")

            x_enc = self._encode(sample)       # (n_modes,)
            out   = np.tanh(x_enc @ self.W)    # (output_dim,)  mirrors Fock-space probs
            features.append(out)

        return np.array(features)


# ── Reservoir config (mirrors QuantumReservoir defaults) ──────────────────────
n_modes       = 6
n_photons     = 2
circuit_depth = 2
n_seeds       = 5

reservoir_features_train = []
reservoir_features_test  = []

for seed in range(n_seeds):
    reservoir = ClassicalReservoir(
        n_modes=n_modes,
        n_photons=n_photons,
        circuit_depth=circuit_depth,
        seed=seed,
    )
    reservoir.fit_scaler(X_train)       # fit on train only — mirrors quantum fit_scaler()

    res_train = reservoir.transform(X_train)
    res_test  = reservoir.transform(X_test)

    reservoir_features_train.append(res_train)
    reservoir_features_test.append(res_test)

# Concatenate all reservoir outputs
reservoir_train = np.hstack(reservoir_features_train)
reservoir_test = np.hstack(reservoir_features_test)

# Concatenate original PCA features (like quantum script)
X_train_full = np.hstack([X_train, reservoir_train])
X_test_full = np.hstack([X_test, reservoir_test])

print("Final feature dimension:", X_train_full.shape[1])

# ==============================
# 3. RIDGE READOUT (MATCH QUANTUM)
# ==============================
# No washout applied — mirrors QuantumReservoir (stateless, no temporal carry-over)

alphas = np.logspace(-6, 3, 20)

ridge = RidgeCV(alphas=alphas)
ridge.fit(X_train_full, y_train)

print("Selected alpha:", ridge.alpha_)

# ==============================
# 5. EVALUATION
# ==============================

predictions      = ridge.predict(X_test_full)
train_predictions = ridge.predict(X_train_full)

test_mse  = mean_squared_error(y_test,  predictions)
train_mse = mean_squared_error(y_train, train_predictions)

print("\nTrain MSE:", train_mse)
print("Test MSE:",  test_mse)
print("Mean absolute target value:", np.mean(np.abs(y_test)))
print("Test variance:", np.var(y_test))
print("Overall Train R2:", r2_score(y_train, train_predictions))
print("Overall Test  R2:", r2_score(y_test,  predictions))

# Per-PC metrics
train_r2_scores, test_r2_scores   = [], []
train_mse_scores, test_mse_scores = [], []
for i in range(6):
    tr2  = r2_score(y_train[:, i], train_predictions[:, i])
    tr2t = r2_score(y_test[:,  i], predictions[:,       i])
    tm   = mean_squared_error(y_train[:, i], train_predictions[:, i])
    tmt  = mean_squared_error(y_test[:,  i], predictions[:,       i])
    train_r2_scores.append(tr2)
    test_r2_scores.append(tr2t)
    train_mse_scores.append(tm)
    test_mse_scores.append(tmt)
    print(f"PC{i+1}  Train R2: {tr2:.4f}  |  Test R2: {tr2t:.4f}")

# Naive baseline
naive_mse = mean_squared_error(y_test, X_test)
print("Naive MSE:", naive_mse)

# ==============================
# 6. SAVE MODEL
# ==============================

joblib.dump(ridge, "classical_ensemble_ridge.pkl")
print("Model saved as classical_ensemble_ridge.pkl")

# ==============================
# 7. GENERATE & SAVE GRAPHS
# ==============================

pc_labels = [f"PC{i+1}" for i in range(6)]
x_idx     = np.arange(6)
bar_width  = 0.35

# ── 7a. Actual vs Predicted (train + test) ──────────────────────
fig, axes = plt.subplots(2, 6, figsize=(20, 8))
fig.suptitle("Actual vs Predicted — Classical Ensemble Reservoir", fontsize=14, fontweight="bold")

for i in range(6):
    # Training row
    ax = axes[0, i]
    ax.scatter(y_train[:, i], train_predictions[:, i], alpha=0.4, s=12, color="steelblue")
    lims = [min(y_train[:, i].min(), train_predictions[:, i].min()),
            max(y_train[:, i].max(), train_predictions[:, i].max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_title(f"{pc_labels[i]} (Train)", fontsize=9)
    ax.set_xlabel("Actual",    fontsize=8)
    ax.set_ylabel("Predicted", fontsize=8)
    ax.annotate(f"R²={train_r2_scores[i]:.3f}", xy=(0.05, 0.90),
                xycoords="axes fraction", fontsize=8)

    # Test row
    ax = axes[1, i]
    ax.scatter(y_test[:, i], predictions[:, i], alpha=0.4, s=12, color="darkorange")
    lims = [min(y_test[:, i].min(), predictions[:, i].min()),
            max(y_test[:, i].max(), predictions[:, i].max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_title(f"{pc_labels[i]} (Test)", fontsize=9)
    ax.set_xlabel("Actual",    fontsize=8)
    ax.set_ylabel("Predicted", fontsize=8)
    ax.annotate(f"R²={test_r2_scores[i]:.3f}", xy=(0.05, 0.90),
                xycoords="axes fraction", fontsize=8)

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: actual_vs_predicted.png")

# ── 7b. R² Score — Train vs Test per PC ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x_idx - bar_width / 2, train_r2_scores, bar_width,
               label="Train", color="steelblue", edgecolor="black")
bars2 = ax.bar(x_idx + bar_width / 2, test_r2_scores,  bar_width,
               label="Test",  color="darkorange", edgecolor="black")
ax.set_xlabel("Principal Component")
ax.set_ylabel("R² Score")
ax.set_title("R² Score per PC — Train vs Test")
ax.set_xticks(x_idx)
ax.set_xticklabels(pc_labels)
ax.legend()
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
plt.tight_layout()
plt.savefig("r2_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: r2_comparison.png")

# ── 7c. MSE per PC — Train vs Test ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x_idx - bar_width / 2, train_mse_scores, bar_width,
       label="Train", color="steelblue", edgecolor="black")
ax.bar(x_idx + bar_width / 2, test_mse_scores,  bar_width,
       label="Test",  color="darkorange", edgecolor="black")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Mean Squared Error")
ax.set_title("MSE per PC — Train vs Test")
ax.set_xticks(x_idx)
ax.set_xticklabels(pc_labels)
ax.legend()
plt.tight_layout()
plt.savefig("mse_per_pc.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: mse_per_pc.png")

# ── 7d. Residuals Distribution per PC (Test Set) ────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Residuals Distribution per PC (Test Set)", fontsize=13, fontweight="bold")
for i, ax in enumerate(axes.flat):
    residuals = y_test[:, i] - predictions[:, i]
    ax.hist(residuals, bins=30, color="mediumpurple", edgecolor="black", alpha=0.8)
    ax.axvline(0,                 color="red",  linewidth=1.5, linestyle="--", label="Zero")
    ax.axvline(residuals.mean(),  color="blue", linewidth=1.2, linestyle="--",
               label=f"Mean={residuals.mean():.3f}")
    ax.set_title(pc_labels[i])
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig("residuals_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: residuals_distribution.png")

# ── 7e. Learning Curve (Train vs Validation MSE vs Training Size) ───
# Uses the best alpha found by RidgeCV; MSE is averaged over 5 CV folds.
best_alpha = ridge.alpha_
cv = KFold(n_splits=5, shuffle=False)

train_sizes, train_scores, val_scores = learning_curve(
    Ridge(alpha=best_alpha),
    X_train_full,
    y_train,
    train_sizes=np.linspace(0.1, 1.0, 15),
    cv=cv,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)

train_loss = -train_scores.mean(axis=1)
val_loss   = -val_scores.mean(axis=1)
train_std  =  train_scores.std(axis=1)
val_std    =  val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_sizes, train_loss, "o-", color="steelblue",   label="Training Loss (MSE)")
ax.plot(train_sizes, val_loss,   "o-", color="darkorange",  label="Validation Loss (MSE)")
ax.fill_between(train_sizes,
                train_loss - train_std, train_loss + train_std,
                alpha=0.15, color="steelblue")
ax.fill_between(train_sizes,
                val_loss - val_std, val_loss + val_std,
                alpha=0.15, color="darkorange")
ax.set_xlabel("Training Set Size")
ax.set_ylabel("MSE")
ax.set_title(f"Learning Curve — Ridge (\u03b1={best_alpha:.2e})")
ax.legend()
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: learning_curve.png")

# ── 7f. Regularization Path (CV Loss vs Alpha) ──────────────────────
# Shows how train and validation MSE change across the alpha search space.
alpha_train_loss = []
alpha_val_loss   = []

for a in alphas:
    _train_sizes, _train_s, _val_s = learning_curve(
        Ridge(alpha=a),
        X_train_full,
        y_train,
        train_sizes=[1.0],
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    alpha_train_loss.append(-_train_s.mean())
    alpha_val_loss.append(-_val_s.mean())

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(alphas, alpha_train_loss, "o-", color="steelblue",  label="Training Loss (MSE)")
ax.semilogx(alphas, alpha_val_loss,   "o-", color="darkorange", label="Validation Loss (MSE)")
ax.axvline(best_alpha, color="red", linestyle="--", linewidth=1.2,
           label=f"Selected \u03b1={best_alpha:.2e}")
ax.set_xlabel("Regularization Strength (α, log scale)")
ax.set_ylabel("MSE")
ax.set_title("Regularization Path — Train vs Validation Loss")
ax.legend()
plt.tight_layout()
plt.savefig("regularization_path.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: regularization_path.png")

print("\nAll graphs saved successfully.")
