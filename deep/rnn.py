"""Recurrent Neural Network cells and sequence model."""

import numpy as np
from typing import Optional, Protocol


class RNNCellBase(Protocol):
    """Protocol for RNN cells."""
    hidden_size: int

    def forward(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        ...


class RNNCell:
    """Vanilla RNN cell with tanh activation.

    Args:
        input_size: Dimensionality of input features.
        hidden_size: Number of hidden units.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        scale_ih = np.sqrt(1.0 / input_size)
        scale_hh = np.sqrt(1.0 / hidden_size)
        self.W_ih = np.random.randn(input_size, hidden_size) * scale_ih
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh
        self.b = np.zeros(hidden_size)

    def forward(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Single time-step forward. x: (batch, input_size), h: (batch, hidden_size)."""
        return np.tanh(x @ self.W_ih + h @ self.W_hh + self.b)


class LSTMCell:
    """LSTM cell with forget, input, and output gates.

    Args:
        input_size: Dimensionality of input features.
        hidden_size: Number of hidden units.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Combined weight matrices for all 4 gates: forget, input, cell, output
        scale = np.sqrt(1.0 / (input_size + hidden_size))
        combined = input_size + hidden_size
        self.W = np.random.randn(combined, 4 * hidden_size) * scale
        self.b = np.zeros(4 * hidden_size)
        # Initialize forget gate bias to 1 for better gradient flow
        self.b[hidden_size:2 * hidden_size] = 1.0

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(
        self, x: np.ndarray, h: np.ndarray, c: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single time-step. Returns (h_new, c_new).

        Args:
            x: Input of shape (batch, input_size).
            h: Hidden state of shape (batch, hidden_size).
            c: Cell state of shape (batch, hidden_size). Zeros if None.
        """
        if c is None:
            c = np.zeros_like(h)
        combined = np.concatenate([x, h], axis=1)
        gates = combined @ self.W + self.b
        hs = self.hidden_size

        i_gate = self._sigmoid(gates[:, :hs])
        f_gate = self._sigmoid(gates[:, hs:2 * hs])
        g_gate = np.tanh(gates[:, 2 * hs:3 * hs])
        o_gate = self._sigmoid(gates[:, 3 * hs:])

        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * np.tanh(c_new)
        return h_new, c_new


class GRUCell:
    """Gated Recurrent Unit cell.

    Args:
        input_size: Dimensionality of input features.
        hidden_size: Number of hidden units.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        scale = np.sqrt(1.0 / (input_size + hidden_size))
        combined = input_size + hidden_size
        # Reset and update gates
        self.W_gates = np.random.randn(combined, 2 * hidden_size) * scale
        self.b_gates = np.zeros(2 * hidden_size)
        # Candidate hidden state
        self.W_candidate = np.random.randn(combined, hidden_size) * scale
        self.b_candidate = np.zeros(hidden_size)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Single time-step. Returns h_new of shape (batch, hidden_size)."""
        combined = np.concatenate([x, h], axis=1)
        gates = self._sigmoid(combined @ self.W_gates + self.b_gates)
        hs = self.hidden_size
        r_gate = gates[:, :hs]  # reset gate
        z_gate = gates[:, hs:]  # update gate

        combined_r = np.concatenate([x, r_gate * h], axis=1)
        h_candidate = np.tanh(combined_r @ self.W_candidate + self.b_candidate)
        h_new = (1 - z_gate) * h + z_gate * h_candidate
        return h_new


class SequenceModel:
    """Wraps an RNN/LSTM/GRU cell for processing sequences.

    Args:
        cell: An RNNCell, LSTMCell, or GRUCell instance.
        output_size: Size of the output projection. None for raw hidden states.
        return_sequences: If True, return all time steps; else return last.
    """

    def __init__(
        self,
        cell: RNNCell | LSTMCell | GRUCell,
        output_size: Optional[int] = None,
        return_sequences: bool = False,
    ) -> None:
        self.cell = cell
        self.return_sequences = return_sequences
        self.output_proj = None
        if output_size is not None:
            scale = np.sqrt(2.0 / cell.hidden_size)
            self.output_proj = {
                "W": np.random.randn(cell.hidden_size, output_size) * scale,
                "b": np.zeros(output_size),
            }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Process a sequence.

        Args:
            x: Input of shape (batch, seq_len, input_size).

        Returns:
            If return_sequences: (batch, seq_len, hidden_size_or_output_size)
            Else: (batch, hidden_size_or_output_size)
        """
        batch_size, seq_len, _ = x.shape
        h = np.zeros((batch_size, self.cell.hidden_size))
        c = np.zeros((batch_size, self.cell.hidden_size)) if isinstance(self.cell, LSTMCell) else None

        outputs = []
        for t in range(seq_len):
            if isinstance(self.cell, LSTMCell):
                h, c = self.cell.forward(x[:, t, :], h, c)
            else:
                h = self.cell.forward(x[:, t, :], h)
            outputs.append(h)

        if self.return_sequences:
            result = np.stack(outputs, axis=1)
        else:
            result = outputs[-1]

        if self.output_proj is not None:
            result = result @ self.output_proj["W"] + self.output_proj["b"]
        return result

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return argmax predictions (for classification)."""
        logits = self.forward(x)
        return np.argmax(logits, axis=-1)
