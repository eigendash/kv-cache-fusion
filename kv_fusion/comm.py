"""The ways a receiver can get an answer, given a question prompt [Q, s1, s2, ?]."""

import mlx.core as mx

from .world import ANS, context_prompt


def _answer_from(receiver, prefix, batch):
    """Decode one step: feed ANS after `prefix` and return logits for the answer."""
    start = mx.full((batch, 1), ANS, dtype=mx.int32)
    logits, _ = receiver(start, prefix=prefix)
    return logits[:, 0]


def receiver_alone(receiver, prompt):
    _, cache = receiver(prompt)
    return _answer_from(receiver, cache, prompt.shape[0])


def sharer_alone(sharer, prompt):
    """The sharer answers straight after the question mark."""
    logits, _ = sharer(prompt)
    return logits[:, -1]


def cache_to_cache(receiver, sharer, fuser, prompt, training=False, sharer_prompt=None):
    """Receiver decodes from a prompt cache that has been fused with the sharer's.

    `sharer_prompt` lets a caller feed the sharer a different question than the receiver
    sees, which is how the mismatched-cache control is run.
    """
    _, receiver_cache = receiver(prompt)
    _, sharer_cache = sharer(prompt if sharer_prompt is None else sharer_prompt)
    sharer_cache = [(mx.stop_gradient(k), mx.stop_gradient(v)) for k, v in sharer_cache]
    receiver_cache = [(mx.stop_gradient(k), mx.stop_gradient(v)) for k, v in receiver_cache]
    fused = fuser(receiver_cache, sharer_cache, training=training)
    return _answer_from(receiver, fused, prompt.shape[0])


def text_to_text(receiver, sharer, entities, prompt):
    """The sharer writes its answer as a token; the receiver reads it as context."""
    guess = mx.argmax(sharer_alone(sharer, prompt), axis=-1)
    ctx = context_prompt(entities, guess)
    _, cache = receiver(ctx)
    return _answer_from(receiver, cache, ctx.shape[0])


def accuracy(logits, targets):
    return float((mx.argmax(logits, axis=-1) == targets).astype(mx.float32).mean())

