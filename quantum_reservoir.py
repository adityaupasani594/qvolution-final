import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from merlin.algorithms import QuantumLayer
from merlin.builder import CircuitBuilder
from merlin.core.state_vector import StateVector
from merlin.measurement import MeasurementStrategy
from merlin.core.computation_space import ComputationSpace

# ── QPU toggle ────────────────────────────────────────────────
# Set to True + provide your Quandela Cloud token to run on Belenos.
# When False the local MerLin statevector simulator is used.
USE_QPU       = False
QPU_TOKEN     = None          # e.g. "your-quandela-cloud-api-token"
QPU_BACKEND   = "qpu:belenos"
QPU_N_SAMPLES = 10_000        # shots per circuit on the QPU


class QuantumReservoir:
    """
    Quantum Optical Reservoir Computing (QORC)
    ------------------------------------------
    - Fixed random interferometer built with MerLin's CircuitBuilder
    - MinMax [0,1] angle encoding  → phase ∈ [0, π]  (no saturation)
    - Fock-space photon-number probability measurement
    - Deterministic feature dimension via MerLin's QuantumLayer
    - Optional remote execution on Quandela Belenos QPU
    """

    def __init__(
        self,
        n_modes=6,
        n_photons=2,
        circuit_depth=2,
        seed=42,
    ):
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.circuit_depth = circuit_depth

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Input scaler — must call fit_scaler() before transform()
        self._input_scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler_fitted = False

        self._layer = self._build_layer()
        self.output_dim = self._layer.output_size

        print(f"Reservoir feature dim: {self.output_dim}")

    # -----------------------------------------------------------
    # Reservoir Construction
    # -----------------------------------------------------------

    def _build_layer(self) -> QuantumLayer:
        builder = CircuitBuilder(n_modes=self.n_modes)

        # Angle encoding: each feature → phase shifter rotation θ = x · π
        # x is pre-scaled to [0, 1] by MinMaxScaler → θ ∈ [0, π]
        builder.add_angle_encoding(
            modes=list(range(self.n_modes)),
            name="input",
            scale=np.pi,
        )

        for _ in range(self.circuit_depth):
            builder.add_entangling_layer(trainable=False)
            builder.add_superpositions(
                modes=list(range(self.n_modes)),
                trainable=False,
                depth=1,
            )

        # |1,1,0,0,0,0⟩ — one photon per first n_photons modes
        input_state_list = [1] * self.n_photons + [0] * (self.n_modes - self.n_photons)

        # ── Processor: local simulator or Belenos QPU ─────────
        processor_kwargs = {}
        if USE_QPU:
            import perceval
            if QPU_TOKEN:
                perceval.set_remote_access_token(QPU_TOKEN)
            remote_proc = perceval.RemoteProcessor(QPU_BACKEND)
            remote_proc.set_parameter("n_samples", QPU_N_SAMPLES)
            processor_kwargs["processor"] = remote_proc
            print(f"  → Using remote QPU: {QPU_BACKEND}  "
                  f"({QPU_N_SAMPLES} shots/circuit)")

        layer = QuantumLayer(
            input_size=self.n_modes,
            builder=builder,
            input_state=StateVector.from_basic_state(input_state_list),
            # Full Fock space: C(n+k-1, k) = 21 output states for (6 modes, 2 photons)
            # probs() is required for SNSPD detectors on the physical QPU
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.FOCK
            ),
            **processor_kwargs,
        )

        for param in layer.parameters():
            param.requires_grad_(False)

        return layer

    # -----------------------------------------------------------
    # Input Scaling
    # -----------------------------------------------------------

    def fit_scaler(self, X_train: np.ndarray):
        """
        Fit the MinMaxScaler on training data so that phase encodings
        use the full [0, π] range without saturation or wrap-around.
        Must be called BEFORE transform().
        """
        self._input_scaler.fit(X_train)
        self._scaler_fitted = True
        return self

    # -----------------------------------------------------------
    # Input Encoding
    # -----------------------------------------------------------

    def _encode(self, x: np.ndarray) -> torch.Tensor:
        if not self._scaler_fitted:
            raise RuntimeError(
                "Input scaler not fitted. Call fit_scaler(X_train) before transform()."
            )
        # MinMax → [0, 1]; builder multiplies by scale=π → θ ∈ [0, π]
        x_scaled = self._input_scaler.transform(x.reshape(1, -1)).flatten()
        return torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)  # (1, n_modes)

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
        features : numpy array of shape (n_samples, reservoir_dim)
        """
        self._layer.eval()
        features = []

        with torch.no_grad():
            for idx, sample in enumerate(X):
                if idx % 50 == 0:
                    print(f"  Processing sample {idx}/{len(X)}...")

                x_tensor = self._encode(sample)       # (1, n_modes)
                probs = self._layer(x_tensor)          # (1, output_dim)
                features.append(probs.squeeze(0).numpy())

        return np.array(features)