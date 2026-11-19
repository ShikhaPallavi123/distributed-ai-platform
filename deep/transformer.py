"""Transformer architecture: encoder, decoder, and full models."""

import numpy as np
from typing import Optional


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization over last dimension."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


class PositionalEncoding:
    """Sinusoidal positional encoding.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        self.encoding = np.zeros((max_len, d_model))
        positions = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = np.sin(positions * div_term)
        self.encoding[:, 1::2] = np.cos(positions * div_term[:d_model // 2])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding. x shape: (batch, seq_len, d_model)."""
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len]


class MultiHeadAttention:
    """Multi-head scaled dot-product attention.

    Args:
        d_model: Total model dimension.
        n_heads: Number of attention heads.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Multi-head attention forward pass.

        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask: Optional broadcastable mask; True positions are masked out.
        """
        batch = query.shape[0]

        Q = (query @ self.W_q).reshape(batch, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = (key @ self.W_k).reshape(batch, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = (value @ self.W_v).reshape(batch, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)

        attn_weights = _softmax(scores, axis=-1)
        context = attn_weights @ V  # (batch, heads, seq_q, d_k)
        context = context.transpose(0, 2, 1, 3).reshape(batch, -1, self.d_model)
        return context @ self.W_o


class FeedForward:
    """Position-wise feed-forward network with ReLU."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x @ self.W1 + self.b1) @ self.W2 + self.b2


class TransformerBlock:
    """Single transformer encoder block: MHA + FFN with residual connections and layer norm.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        attn_out = self.attention.forward(x, x, x, mask)
        x = _layer_norm(x + attn_out, self.ln1_gamma, self.ln1_beta)
        ff_out = self.ff.forward(x)
        x = _layer_norm(x + ff_out, self.ln2_gamma, self.ln2_beta)
        return x


class TransformerEncoder:
    """Stack of N transformer encoder blocks.

    Args:
        n_layers: Number of transformer blocks.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        max_len: Max sequence length for positional encoding.
    """

    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, max_len: int = 5000
    ) -> None:
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Encode input. x: (batch, seq_len, d_model)."""
        x = self.pos_encoding.forward(x)
        for block in self.blocks:
            x = block.forward(x, mask)
        return x


class TransformerDecoderBlock:
    """Single transformer decoder block with masked self-attention and cross-attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)
        self.ln3_gamma = np.ones(d_model)
        self.ln3_beta = np.zeros(d_model)

    @staticmethod
    def _causal_mask(seq_len: int) -> np.ndarray:
        """Upper triangular mask: True where attention is blocked."""
        return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

    def forward(
        self,
        x: np.ndarray,
        encoder_out: np.ndarray,
        src_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        seq_len = x.shape[1]
        causal = self._causal_mask(seq_len)
        # Masked self-attention
        sa_out = self.self_attn.forward(x, x, x, causal)
        x = _layer_norm(x + sa_out, self.ln1_gamma, self.ln1_beta)
        # Cross-attention over encoder output
        ca_out = self.cross_attn.forward(x, encoder_out, encoder_out, src_mask)
        x = _layer_norm(x + ca_out, self.ln2_gamma, self.ln2_beta)
        # Feed-forward
        ff_out = self.ff.forward(x)
        x = _layer_norm(x + ff_out, self.ln3_gamma, self.ln3_beta)
        return x


class TransformerDecoder:
    """Stack of N transformer decoder blocks."""

    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, max_len: int = 5000
    ) -> None:
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = [TransformerDecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]

    def forward(
        self,
        x: np.ndarray,
        encoder_out: np.ndarray,
        src_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        x = self.pos_encoding.forward(x)
        for block in self.blocks:
            x = block.forward(x, encoder_out, src_mask)
        return x


class Transformer:
    """Full encoder-decoder transformer model.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_enc_layers: Number of encoder layers.
        n_dec_layers: Number of decoder layers.
        d_ff: Feed-forward hidden dimension.
        max_len: Maximum sequence length.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_enc_layers: int = 6,
        n_dec_layers: int = 6,
        d_ff: int = 1024,
        max_len: int = 512,
    ) -> None:
        self.d_model = d_model
        self.src_embed = np.random.randn(src_vocab_size, d_model) * 0.02
        self.tgt_embed = np.random.randn(tgt_vocab_size, d_model) * 0.02
        self.encoder = TransformerEncoder(n_enc_layers, d_model, n_heads, d_ff, max_len)
        self.decoder = TransformerDecoder(n_dec_layers, d_model, n_heads, d_ff, max_len)
        scale = np.sqrt(2.0 / d_model)
        self.output_proj = np.random.randn(d_model, tgt_vocab_size) * scale

    def forward(
        self,
        src_tokens: np.ndarray,
        tgt_tokens: np.ndarray,
        src_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward pass. src/tgt_tokens: (batch, seq_len) of integer indices.

        Returns logits of shape (batch, tgt_seq_len, tgt_vocab_size).
        """
        src_emb = self.src_embed[src_tokens] * np.sqrt(self.d_model)
        tgt_emb = self.tgt_embed[tgt_tokens] * np.sqrt(self.d_model)
        enc_out = self.encoder.forward(src_emb, src_mask)
        dec_out = self.decoder.forward(tgt_emb, enc_out, src_mask)
        return dec_out @ self.output_proj


class GPTModel:
    """Decoder-only transformer for autoregressive text generation.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        d_ff: Feed-forward hidden dimension.
        max_len: Maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_len: int = 512,
    ) -> None:
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_embed = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_gamma = np.ones(d_model)
        self.ln_beta = np.zeros(d_model)
        scale = np.sqrt(2.0 / d_model)
        self.output_proj = np.random.randn(d_model, vocab_size) * scale

    @staticmethod
    def _causal_mask(seq_len: int) -> np.ndarray:
        return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """Forward pass. tokens: (batch, seq_len) integer indices.

        Returns logits of shape (batch, seq_len, vocab_size).
        """
        x = self.token_embed[tokens] * np.sqrt(self.d_model)
        x = self.pos_encoding.forward(x)
        mask = self._causal_mask(tokens.shape[1])
        for block in self.blocks:
            x = block.forward(x, mask)
        x = _layer_norm(x, self.ln_gamma, self.ln_beta)
        return x @ self.output_proj

    def generate(
        self,
        prompt_tokens: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Autoregressively generate tokens.

        Args:
            prompt_tokens: (1, prompt_len) integer indices.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (lower = more greedy).

        Returns:
            (1, prompt_len + max_new_tokens) integer array.
        """
        tokens = np.array(prompt_tokens, dtype=int)
        for _ in range(max_new_tokens):
            logits = self.forward(tokens)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = _softmax(next_logits, axis=-1)
            next_token = np.array([[np.random.choice(self.vocab_size, p=probs[0])]])
            tokens = np.concatenate([tokens, next_token], axis=1)
        return tokens
