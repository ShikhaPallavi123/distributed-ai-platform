"""Text generation module with transformer-like architecture, sampling strategies, and conversational AI.

Implements character/word-level tokenization, self-attention, and various decoding
strategies including temperature sampling, top-k, nucleus (top-p), and beam search.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np


class TokenizerMode(Enum):
    """Tokenization granularity."""
    CHAR = "char"
    WORD = "word"


class Tokenizer:
    """Character-level and word-level tokenizer with vocabulary management."""

    SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

    def __init__(self, mode: TokenizerMode = TokenizerMode.WORD) -> None:
        self.mode = mode
        self.token_to_id: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self._next_id = len(self.SPECIAL_TOKENS)

    def build_vocab(self, corpus: str, min_freq: int = 1) -> None:
        """Build vocabulary from a text corpus."""
        tokens = self._tokenize(corpus)
        counts = Counter(tokens)
        for token, freq in sorted(counts.items()):
            if freq >= min_freq and token not in self.token_to_id:
                self.token_to_id[token] = self._next_id
                self.id_to_token[self._next_id] = token
                self._next_id += 1

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _tokenize(self, text: str) -> List[str]:
        if self.mode == TokenizerMode.CHAR:
            return list(text)
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = self._tokenize(text)
        ids = [self.token_to_id.get(t, self.SPECIAL_TOKENS["<unk>"]) for t in tokens]
        if add_special:
            ids = [self.SPECIAL_TOKENS["<bos>"]] + ids + [self.SPECIAL_TOKENS["<eos>"]]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for tid in ids:
            tok = self.id_to_token.get(tid, "<unk>")
            if skip_special and tok in self.SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        sep = "" if self.mode == TokenizerMode.CHAR else " "
        return sep.join(tokens)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


class TextGenerator:
    """Transformer-like text generator with self-attention and multiple sampling strategies.

    Uses a simplified architecture: token embedding, positional encoding,
    stacked self-attention layers, and a language-model head.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 128,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.rng = np.random.RandomState(seed)
        vocab = tokenizer.vocab_size
        scale = 0.02

        # Token embeddings and positional encoding
        self.token_emb = self.rng.randn(vocab, d_model).astype(np.float32) * scale
        self.pos_enc = self._sinusoidal_pe(max_seq_len, d_model)

        # Attention weights per layer: Wq, Wk, Wv, Wo
        self.layers: List[Dict[str, np.ndarray]] = []
        for _ in range(n_layers):
            self.layers.append({
                "Wq": self.rng.randn(d_model, d_model).astype(np.float32) * scale,
                "Wk": self.rng.randn(d_model, d_model).astype(np.float32) * scale,
                "Wv": self.rng.randn(d_model, d_model).astype(np.float32) * scale,
                "Wo": self.rng.randn(d_model, d_model).astype(np.float32) * scale,
            })

        # Language model head
        self.lm_head = self.rng.randn(d_model, vocab).astype(np.float32) * scale

    @staticmethod
    def _sinusoidal_pe(length: int, d_model: int) -> np.ndarray:
        pe = np.zeros((length, d_model), dtype=np.float32)
        pos = np.arange(length)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        return pe

    def _self_attention(self, x: np.ndarray, layer: Dict[str, np.ndarray]) -> np.ndarray:
        """Multi-head causal self-attention."""
        seq_len, _ = x.shape
        head_dim = self.d_model // self.n_heads
        Q = x @ layer["Wq"]
        K = x @ layer["Wk"]
        V = x @ layer["Wv"]

        # Reshape for multi-head
        Q = Q.reshape(seq_len, self.n_heads, head_dim)
        K = K.reshape(seq_len, self.n_heads, head_dim)
        V = V.reshape(seq_len, self.n_heads, head_dim)

        # Scaled dot-product attention with causal mask
        scores = np.einsum("qhd,khd->hqk", Q, K) / math.sqrt(head_dim)
        causal_mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
        scores += causal_mask[None, :, :]
        attn = _softmax(scores)
        out = np.einsum("hqk,khd->qhd", attn, V)
        out = out.reshape(seq_len, self.d_model)
        return out @ layer["Wo"]

    def _forward(self, token_ids: List[int]) -> np.ndarray:
        """Forward pass returning logits for next token at each position."""
        seq_len = min(len(token_ids), self.max_seq_len)
        ids = token_ids[-seq_len:]
        x = self.token_emb[ids] + self.pos_enc[:seq_len]
        for layer in self.layers:
            residual = x
            x = _layer_norm(self._self_attention(x, layer) + residual)
        logits = x @ self.lm_head  # (seq_len, vocab)
        return logits

    def _sample_token(
        self, logits: np.ndarray, temperature: float, top_k: int, top_p: float
    ) -> int:
        """Sample a token from logits using temperature, top-k, and nucleus (top-p)."""
        logits = logits / max(temperature, 1e-8)

        if top_k > 0:
            indices = np.argsort(logits)[::-1]
            cutoff = logits[indices[min(top_k, len(indices) - 1)]]
            logits[logits < cutoff] = -1e9

        probs = _softmax(logits)

        if 0.0 < top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cum = np.cumsum(probs[sorted_idx])
            mask_idx = sorted_idx[cum > top_p]
            if len(mask_idx) > 0 and len(mask_idx) < len(probs):
                probs[mask_idx] = 0.0
            probs /= probs.sum()

        return int(self.rng.choice(len(probs), p=probs))

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        beam_width: int = 0,
    ) -> str:
        """Generate text continuation from a prompt.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            top_k: Top-k filtering (0 to disable).
            top_p: Nucleus sampling threshold (1.0 to disable).
            beam_width: If > 0, use beam search instead of sampling.

        Returns:
            Generated text string.
        """
        ids = self.tokenizer.encode(prompt, add_special=True)

        if beam_width > 0:
            ids = self._beam_search(ids, max_tokens, beam_width)
        else:
            for _ in range(max_tokens):
                logits = self._forward(ids)
                next_logits = logits[-1]
                tok = self._sample_token(next_logits, temperature, top_k, top_p)
                ids.append(tok)
                if tok == Tokenizer.SPECIAL_TOKENS["<eos>"]:
                    break

        return self.tokenizer.decode(ids, skip_special=True)

    def _beam_search(
        self, initial_ids: List[int], max_tokens: int, beam_width: int
    ) -> List[int]:
        """Beam search decoding for higher-quality outputs."""
        beams: List[Tuple[List[int], float]] = [(initial_ids, 0.0)]
        eos = Tokenizer.SPECIAL_TOKENS["<eos>"]

        for _ in range(max_tokens):
            candidates: List[Tuple[List[int], float]] = []
            for seq, score in beams:
                if seq[-1] == eos:
                    candidates.append((seq, score))
                    continue
                logits = self._forward(seq)[-1]
                log_probs = np.log(_softmax(logits) + 1e-10)
                top_indices = np.argsort(log_probs)[-beam_width:]
                for idx in top_indices:
                    candidates.append((seq + [int(idx)], score + log_probs[idx]))

            beams = sorted(candidates, key=lambda b: b[1], reverse=True)[:beam_width]
            if all(b[0][-1] == eos for b in beams):
                break

        return beams[0][0]


class ConversationalAI:
    """Conversational wrapper maintaining a sliding context window over a TextGenerator."""

    def __init__(self, generator: TextGenerator, context_window: int = 512) -> None:
        self.generator = generator
        self.context_window = context_window
        self.history: List[Dict[str, str]] = []

    def _build_prompt(self) -> str:
        parts: List[str] = []
        for turn in self.history:
            parts.append(f"User: {turn['user']}")
            if turn.get("assistant"):
                parts.append(f"Assistant: {turn['assistant']}")
        return "\n".join(parts) + "\nAssistant:"

    def chat(self, user_message: str, **gen_kwargs) -> str:
        """Send a user message and get a response, maintaining conversation context."""
        self.history.append({"user": user_message, "assistant": ""})
        prompt = self._build_prompt()

        # Trim context to window
        encoded = self.generator.tokenizer.encode(prompt, add_special=False)
        if len(encoded) > self.context_window:
            # Drop oldest turns until it fits
            while len(encoded) > self.context_window and len(self.history) > 1:
                self.history.pop(0)
                prompt = self._build_prompt()
                encoded = self.generator.tokenizer.encode(prompt, add_special=False)

        response = self.generator.generate(prompt, **gen_kwargs)
        # Extract only the new part after the prompt
        if response.startswith(prompt.lower().rstrip()):
            response = response[len(prompt.lower().rstrip()):].strip()
        self.history[-1]["assistant"] = response
        return response

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()
