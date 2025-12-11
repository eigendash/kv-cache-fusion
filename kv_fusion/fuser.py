"""Cache fuser: map a sharer's KV-cache into a receiver's and add it residually.

For every receiver layer n the fuser reads the receiver's cache and the cache of a
paired sharer layer G(n), and produces

    fused_n = receiver_n + gate_n * weights(x) * fuse(project([receiver_n ; sharer_G(n)]))

where x is the concatenated input. Keys and values get separate fusers. The gate is a
learned scalar per layer: a Gumbel-sigmoid with a temperature that is annealed during
training, and a hard threshold at inference. Both models stay frozen; only the fuser
is trained.
"""

import mlx.core as mx
import mlx.nn as nn


def terminal_layer_map(n_receiver, n_sharer):
    """Pair layers from the output end: last with last, penultimate with penultimate, ..."""
    return [n_sharer - n_receiver + n for n in range(n_receiver)]


def depth_layer_map(n_receiver, n_sharer):
    """Pair layers by relative depth, so correspondences spread over the whole stack."""
    if n_receiver == 1:
        return [n_sharer - 1]
    return [round(n * (n_sharer - 1) / (n_receiver - 1)) for n in range(n_receiver)]


def gumbel_sigmoid(logit, tau, training):
    if not training:
        return (logit > 0).astype(mx.float32)
    u = mx.random.uniform(shape=logit.shape, low=1e-6, high=1 - 1e-6)
    return mx.sigmoid((logit + mx.log(u) - mx.log1p(-u)) / tau)


class LayerFuser(nn.Module):
    def __init__(self, d_receiver, d_sharer, heads, hidden):
        super().__init__()
        d_in = d_receiver + d_sharer
        self.heads = heads
        self.project = nn.Linear(d_in, hidden)
        self.fuse = nn.Linear(hidden, d_receiver)
        self.fuse.weight = self.fuse.weight * 0.1
        self.head_weight = nn.Linear(d_in, heads)
        self.gate_logit = mx.array(0.0)

    def delta(self, receiver, sharer):
        x = mx.concatenate([receiver, sharer], axis=-1)
        f = self.fuse(nn.gelu(self.project(x)))
        B, T, _ = f.shape
        w = mx.sigmoid(self.head_weight(x))[..., None]  # input-aware weight per head
        return (f.reshape(B, T, self.heads, -1) * w).reshape(B, T, -1)

    def __call__(self, receiver, sharer, tau, training):
        gate = gumbel_sigmoid(self.gate_logit, tau, training)
        return receiver + gate * self.delta(receiver, sharer)


def _flat(x):  # (B, H, T, dh) -> (B, T, H*dh)
    B, H, T, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * dh)


def _heads(x, heads):  # (B, T, H*dh) -> (B, H, T, dh)
    B, T, D = x.shape
    return x.reshape(B, T, heads, D // heads).transpose(0, 2, 1, 3)


class CacheFuser(nn.Module):
    def __init__(self, receiver_cfg, sharer_cfg, hidden=128, layer_map=terminal_layer_map):
        super().__init__()
        self.heads = receiver_cfg.heads
        self.map = layer_map(receiver_cfg.layers, sharer_cfg.layers)
        d_r, d_s = receiver_cfg.dim, sharer_cfg.dim
        self.key = [LayerFuser(d_r, d_s, self.heads, hidden) for _ in range(receiver_cfg.layers)]
        self.value = [LayerFuser(d_r, d_s, self.heads, hidden) for _ in range(receiver_cfg.layers)]
        self.tau = 1.0

    def __call__(self, receiver_caches, sharer_caches, training=False):
        fused = []
        for n, (rk, rv) in enumerate(receiver_caches):
            sk, sv = sharer_caches[self.map[n]]
            k = self.key[n](_flat(rk), _flat(sk), self.tau, training)
            v = self.value[n](_flat(rv), _flat(sv), self.tau, training)
            fused.append((_heads(k, self.heads), _heads(v, self.heads)))
        return fused

    def gates(self):
        """Hard gate decisions per layer, keys then values."""
        return [(float(k.gate_logit) > 0, float(v.gate_logit) > 0) for k, v in zip(self.key, self.value)]


def anneal(step, total, start=1.0, end=0.1):
    """Geometric temperature schedule from `start` to `end`."""
    frac = min(1.0, step / max(1, total))
    return start * (end / start) ** frac
