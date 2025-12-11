import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .comm import cache_to_cache
from .fuser import anneal
from .world import fuser_batch, lm_batch


def train_lm(model, world, role, steps, batch=128, lr=2e-3, seed=0, log=None):
    """Next-token training on masked value positions."""
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=optim.cosine_decay(lr, steps), weight_decay=0.0)

    def loss_fn(model, tokens, mask):
        logits, _ = model(tokens[:, :-1])
        ce = nn.losses.cross_entropy(logits, tokens[:, 1:])
        return (ce * mask).sum() / mask.sum()

    step = nn.value_and_grad(model, loss_fn)
    for i in range(1, steps + 1):
        tokens, mask = lm_batch(world, role, rng, batch)
        loss, grads = step(model, tokens, mask)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if log and i % 500 == 0:
            log(f"  {role} step {i:5d}  loss {float(loss):.4f}")
    return model


def train_fuser(fuser, receiver, sharer, world, steps, batch=128, lr=2e-3, seed=0, log=None):
    """Train only the fuser: the receiver predicts the answer from the fused cache."""
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=optim.cosine_decay(lr, steps), weight_decay=0.0)

    def loss_fn(fuser, prompt, target):
        logits = cache_to_cache(receiver, sharer, fuser, prompt, training=True)
        return nn.losses.cross_entropy(logits, target).mean()

    step = nn.value_and_grad(fuser, loss_fn)
    for i in range(1, steps + 1):
        fuser.tau = anneal(i, steps)
        prompt, target = fuser_batch(world, rng, batch)
        loss, grads = step(fuser, prompt, target)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(fuser, grads)
        mx.eval(fuser.parameters(), opt.state, loss)
        if log and i % 500 == 0:
            log(f"  fuser step {i:5d}  loss {float(loss):.4f}  tau {fuser.tau:.2f}")
    return fuser
