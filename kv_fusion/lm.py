"""A small causal transformer that exposes its per-layer key/value cache.

Positions are learned absolute embeddings added at the input, so cached keys carry no
rotary phase. That keeps a cache from another model easy to add to this one. The cache
is returned per layer as (K, V), each shaped (batch, heads, tokens, head_dim).
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class LMConfig:
    vocab: int
    dim: int
    heads: int
    layers: int
    max_len: int = 32

    @property
    def head_dim(self):
        return self.dim // self.heads


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def __call__(self, x, prefix, mask):
        B, T, D = x.shape
        q, k, v = (t.reshape(B, T, self.heads, -1).transpose(0, 2, 1, 3) for t in mx.split(self.qkv(self.ln1(x)), 3, axis=-1))
        keys, values = (k, v) if prefix is None else (mx.concatenate([prefix[0], k], 2), mx.concatenate([prefix[1], v], 2))
        o = mx.fast.scaled_dot_product_attention(q, keys, values, scale=q.shape[-1] ** -0.5, mask=mask)
        x = x + self.proj(o.transpose(0, 2, 1, 3).reshape(B, T, D))
        return x + self.mlp(self.ln2(x)), (k, v)


class TinyLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.dim)
        self.pos = nn.Embedding(cfg.max_len, cfg.dim)
        self.blocks = [Block(cfg.dim, cfg.heads) for _ in range(cfg.layers)]
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab, bias=False)

    def __call__(self, tokens, prefix=None):
        """Run `tokens` after an optional per-layer cache `prefix`.

        Returns (logits, caches) where caches hold only the new tokens' keys and values.
        New tokens attend to every prefix position and causally to each other.
        """
        T = tokens.shape[1]
        past = 0 if prefix is None else prefix[0][0].shape[2]
        mask = (mx.arange(T)[:, None] + past) >= mx.arange(past + T)[None, :]
        x = self.embed(tokens) + self.pos(mx.arange(past, past + T))
        caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, None if prefix is None else prefix[i], mask)
            caches.append(cache)
        return self.head(self.ln_f(x)), caches
