"""A small synthetic world of facts, split between the two models.

Each entity is a pair of syllable tokens and carries one attribute value. The sharer
is trained on facts about set A and the receiver on set B, so neither can answer
questions about the other's entities from its own weights. Set A is split again: the
fuser trains on A_train questions and is scored on A_test, entities the sharer knows
but the fuser has never been asked about. A fuser that only memorised entity-to-value
pairs would fail there.

Token layout: PAD, F (fact), Q (question), ?, ANS, EOS, then syllables, then values.
"""

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

PAD, FACT, QUESTION, QMARK, ANS, EOS = range(6)
N_SYLLABLES, N_VALUES = 16, 24
SYLLABLE0 = 6
VALUE0 = SYLLABLE0 + N_SYLLABLES
VOCAB = VALUE0 + N_VALUES
SEQ_LEN = 10


@dataclass
class World:
    attribute: np.ndarray  # value index per entity
    sharer: np.ndarray  # entities the sharer is trained on (A)
    receiver: np.ndarray  # entities the receiver is trained on (B)
    fuser_train: np.ndarray  # subset of A
    fuser_test: np.ndarray  # rest of A

    @staticmethod
    def make(seed=0, n_a=128, n_b=128, n_test=32):
        n = N_SYLLABLES * N_SYLLABLES
        assert n_a + n_b <= n
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        a, b = order[:n_a], order[n_a : n_a + n_b]
        return World(
            attribute=rng.integers(0, N_VALUES, n),
            sharer=a,
            receiver=b,
            fuser_train=a[: n_a - n_test],
            fuser_test=a[n_a - n_test :],
        )

    @property
    def n_entities(self):
        return N_SYLLABLES * N_SYLLABLES


def entity_tokens(e):
    return np.stack([SYLLABLE0 + e // N_SYLLABLES, SYLLABLE0 + e % N_SYLLABLES], axis=-1)


def value_token(v):
    return VALUE0 + v


def question_prompt(entities):
    """[Q, s1, s2, ?] for each entity, as an mx array."""
    e = np.asarray(entities)
    q = np.full((len(e), 1), QUESTION)
    m = np.full((len(e), 1), QMARK)
    return mx.array(np.concatenate([q, entity_tokens(e), m], axis=1).astype(np.int32))


def _pad(rows):
    out = np.full((len(rows), SEQ_LEN), PAD, dtype=np.int32)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def _example(kind, e, value):
    s1, s2 = entity_tokens(e)
    v = value_token(value)
    # Each example returns its tokens and the index of the value token to be predicted.
    if kind == "fact":  # F s1 s2 v
        return [FACT, s1, s2, v], [3]
    if kind == "ask":  # Q s1 s2 ? v        (the sharer answers straight after the ?)
        return [QUESTION, s1, s2, QMARK, v], [4]
    if kind == "ask_ans":  # Q s1 s2 ? ANS v   (the receiver answers after ANS)
        return [QUESTION, s1, s2, QMARK, ANS, v], [5]
    if kind == "context":  # F s1 s2 v Q s1 s2 ? ANS v, with v drawn at random
        return [FACT, s1, s2, v, QUESTION, s1, s2, QMARK, ANS, v], [9]
    raise ValueError(kind)


def lm_batch(world, role, rng, batch):
    """Training tokens and a loss mask over target positions for a base model.

    Only positions that predict a *known* value are supervised. For in-context
    examples the value is random, so only the answer after reading it is scored.
    """
    rows, targets = [], []
    for _ in range(batch):
        if role == "sharer":
            e = rng.choice(world.sharer)
            kind = rng.choice(["fact", "ask"])
            seq, pos = _example(kind, e, world.attribute[e])
        else:
            kind = rng.choice(["fact", "ask_ans", "context"], p=[0.3, 0.3, 0.4])
            if kind == "context":
                e = rng.integers(0, world.n_entities)
                seq, pos = _example(kind, e, int(rng.integers(0, N_VALUES)))
            else:
                e = rng.choice(world.receiver)
                seq, pos = _example(kind, e, world.attribute[e])
        rows.append(seq)
        targets.append(pos)
    tokens = _pad(rows)
    mask = np.zeros((batch, SEQ_LEN - 1), dtype=np.float32)
    for i, pos in enumerate(targets):
        for p in pos:
            mask[i, p - 1] = 1.0  # target index p-1 predicts token p
    return mx.array(tokens), mx.array(mask)


def fuser_batch(world, rng, batch, receiver_fraction=0.0):
    """Questions for fuser training.

    By default every question is about the sharer's facts. With `receiver_fraction` > 0
    that share of the batch asks about the receiver's own facts instead, which the
    sharer cannot answer, so the fuser has to learn when not to inject anything.
    """
    n_own = int(round(batch * receiver_fraction))
    entities = np.concatenate([rng.choice(world.fuser_train, batch - n_own), rng.choice(world.receiver, n_own)])
    return question_prompt(entities), mx.array(value_token(world.attribute[entities]).astype(np.int32))


def context_prompt(entities, values):
    """Text-to-text prompt: [F, s1, s2, v, Q, s1, s2, ?] built from a sharer-supplied value."""
    e, v = np.asarray(entities), np.asarray(values)
    f = np.full((len(e), 1), FACT)
    q = np.full((len(e), 1), QUESTION)
    m = np.full((len(e), 1), QMARK)
    toks = np.concatenate([f, entity_tokens(e), v[:, None], q, entity_tokens(e), m], axis=1)
    return mx.array(toks.astype(np.int32))


def question_targets(world, entities):
    """Prompt and answer tokens for a set of entities."""
    e = np.asarray(entities)
    return question_prompt(e), mx.array(value_token(world.attribute[e]).astype(np.int32))
