import numpy as np
import os
import torch
from math import comb
from sklearn.preprocessing import MinMaxScaler
from merlin.algorithms import QuantumLayer
from merlin.builder import CircuitBuilder
from merlin.core.state_vector import StateVector
from merlin.measurement import MeasurementStrategy
from merlin.core.computation_space import ComputationSpace

# ── QPU toggle ────────────────────────────────────────────────
# Set to True + provide your Quandela Cloud token to run on Belenos.
# When False the local MerLin statevector simulator is used.
USE_QPU       = True
QPU_TOKEN     = os.environ.get('QUANDELA_TOKEN') # e.g. "your-quandela-cloud-api-token"
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
        use_qpu=None,
        qpu_token=None,
        qpu_backend=None,
        qpu_n_samples=None,
    ):
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.circuit_depth = circuit_depth
        # QPU / remote processor configuration (can be passed or read from env)
        self._use_qpu = USE_QPU if use_qpu is None else bool(use_qpu)
        # token precedence: explicit arg -> env var QUANDELA_TOKEN -> module QPU_TOKEN
        self._qpu_token = qpu_token or os.environ.get("QUANDELA_TOKEN") or QPU_TOKEN
        self._qpu_backend = qpu_backend or QPU_BACKEND
        self._qpu_n_samples = qpu_n_samples or QPU_N_SAMPLES

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Input scaler — must call fit_scaler() before transform()
        self._input_scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler_fitted = False

        # Runtime objects (local layer OR remote processor/sampler)
        self._layer = None
        self._remote_processor = None
        self._remote_sampler = None
        self._basis_index = None
        self._input_param_names = [f"input{i+1}" for i in range(self.n_modes)]

        self._layer = self._build_layer()
        if self._use_qpu:
            self.output_dim = comb(self.n_modes + self.n_photons - 1, self.n_photons)
        else:
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

        if self._use_qpu:
            import perceval as pcvl
            from perceval.algorithm import Sampler

            if not self._qpu_token:
                raise ValueError(
                    "QPU token missing. Set QUANDELA_TOKEN or pass qpu_token when use_qpu=True."
                )

            circuit = builder.to_pcvl_circuit()
            self._remote_processor = pcvl.RemoteProcessor(self._qpu_backend, token=self._qpu_token)
            self._remote_processor.set_circuit(circuit)
            self._remote_processor.with_input(pcvl.BasicState(input_state_list))
            self._remote_processor.min_detected_photons_filter(self.n_photons)
            self._remote_processor.set_parameter("n_samples", self._qpu_n_samples)
            self._remote_sampler = Sampler(
                self._remote_processor,
                max_shots_per_call=self._qpu_n_samples,
            )

            # Basis mapping for deterministic feature ordering
            basis = self._generate_fock_basis(self.n_modes, self.n_photons)
            self._basis_index = {state: i for i, state in enumerate(basis)}

            masked = self._qpu_token[:4] + "..." + self._qpu_token[-4:]
            print(f"  → Using remote QPU: {self._qpu_backend}  ({self._qpu_n_samples} shots/circuit)")
            print(f"  → Token loaded (masked): {masked}")
            return None

        layer = QuantumLayer(
            input_size=self.n_modes,
            builder=builder,
            input_state=StateVector.from_basic_state(input_state_list),
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.FOCK
            ),
        )

        for param in layer.parameters():
            param.requires_grad_(False)

        return layer

    @staticmethod
    def _generate_fock_basis(n_modes: int, n_photons: int):
        basis = []

        def rec(remaining_modes, remaining_photons, prefix):
            if remaining_modes == 1:
                basis.append(tuple(prefix + [remaining_photons]))
                return
            for value in range(remaining_photons + 1):
                rec(remaining_modes - 1, remaining_photons - value, prefix + [value])

        rec(n_modes, n_photons, [])
        return basis

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
        if not self._scaler_fitted:
            raise RuntimeError(
                "Input scaler not fitted. Call fit_scaler(X_train) before transform()."
            )

        if self._use_qpu:
            features = []
            for idx, sample in enumerate(X):
                if idx % 50 == 0:
                    print(f"  Processing sample {idx}/{len(X)}...")

                x_scaled = self._input_scaler.transform(sample.reshape(1, -1)).flatten()
                params = {name: float(value) for name, value in zip(self._input_param_names, x_scaled)}
                self._remote_processor.set_parameters(params)


                result = self._remote_sampler.probs.execute_sync()
                if result is None:
                    print("[QPU ERROR] Remote job returned None. This usually means the job failed, was rejected, or timed out.")
                    print("  Check your Quandela dashboard for job status, quota, or errors.")
                    raise RuntimeError("QPU job failed or returned no result.")
                probs_dict = result.get("results", {})

                vec = np.zeros(self.output_dim, dtype=float)
                for state, prob in probs_dict.items():
                    key = tuple(state)
                    pos = self._basis_index.get(key)
                    if pos is not None:
                        vec[pos] = float(prob)
                features.append(vec)

            return np.array(features)

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

    def smoke_test(self, sample: np.ndarray = None) -> np.ndarray:
        """Run a quick smoke test through the reservoir.

        If `sample` is None, uses the mid-point encoding (0.5) across modes.
        Requires `fit_scaler()` to have been called.
        Returns the reservoir feature vector for the provided sample.
        """
        if not self._scaler_fitted:
            raise RuntimeError(
                "Input scaler not fitted. Call fit_scaler(X_train) before smoke_test()."
            )
        if sample is None:
            sample = np.ones(self.n_modes) * 0.5
        sample = np.asarray(sample).reshape(1, -1)
        return self.transform(sample)