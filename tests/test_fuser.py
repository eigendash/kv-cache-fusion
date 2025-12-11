import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from kv_fusion import CacheFuser, LMConfig, TinyLM, cache_to_cache, depth_layer_map, receiver_alone, terminal_layer_map
from kv_fusion.fuser import anneal, gumbel_sigmoid
from kv_fusion.world import ANS, VOCAB, question_prompt

R_CFG = LMConfig(vocab=VOCAB, dim=32, heads=4, layers=2)
S_CFG = LMConfig(vocab=VOCAB, dim=48, heads=4, layers=5)


def models():
    mx.random.seed(0)
    return TinyLM(R_CFG), TinyLM(S_CFG), CacheFuser(R_CFG, S_CFG, hidden=32)


def test_terminal_alignment_pairs_from_the_output_end():
    assert terminal_layer_map(3, 6) == [3, 4, 5]
    assert terminal_layer_map(4, 4) == [0, 1, 2, 3]


def test_depth_alignment_spreads_over_the_stack():
    assert depth_layer_map(3, 5) == [0, 2, 4]
    assert depth_layer_map(1, 7) == [6]


def test_fused_cache_has_the_receivers_shape():
    receiver, sharer, fuser = models()
    prompt = question_prompt([3, 9, 30])
    _, rc = receiver(prompt)
    _, sc = sharer(prompt)
    fused = fuser(rc, sc)
    assert [k.shape for k, _ in fused] == [k.shape for k, _ in rc]
    assert [v.shape for _, v in fused] == [v.shape for _, v in rc]


def test_closed_gate_returns_the_receivers_own_cache():
    receiver, sharer, fuser = models()
    for layer in fuser.key + fuser.value:
        layer.gate_logit = mx.array(-5.0)
    prompt = question_prompt([1, 2])
    _, rc = receiver(prompt)
    _, sc = sharer(prompt)
    for (fk, fv), (rk, rv) in zip(fuser(rc, sc), rc):
        np.testing.assert_allclose(np.array(fk), np.array(rk), atol=1e-6)
        np.testing.assert_allclose(np.array(fv), np.array(rv), atol=1e-6)


def test_closed_gate_c2c_equals_receiver_alone_and_a_plain_forward_pass():
    receiver, sharer, fuser = models()
    for layer in fuser.key + fuser.value:
        layer.gate_logit = mx.array(-5.0)
    prompt = question_prompt([5, 6, 7])
    solo = receiver_alone(receiver, prompt)
    fused = cache_to_cache(receiver, sharer, fuser, prompt)
    full, _ = receiver(mx.concatenate([prompt, mx.full((3, 1), ANS, dtype=mx.int32)], axis=1))
    np.testing.assert_allclose(np.array(fused), np.array(solo), atol=1e-5)
    np.testing.assert_allclose(np.array(solo), np.array(full[:, -1]), atol=1e-5)


def test_open_gate_actually_uses_the_sharer():
    receiver, sharer, fuser = models()
    for layer in fuser.key + fuser.value:
        layer.gate_logit = mx.array(5.0)
    a, b = question_prompt([3, 4]), question_prompt([200, 201])
    same = cache_to_cache(receiver, sharer, fuser, a)
    other = cache_to_cache(receiver, sharer, fuser, a, sharer_prompt=b)
    assert not np.allclose(np.array(same), np.array(other), atol=1e-4)


def test_only_the_fuser_receives_gradients():
    receiver, sharer, fuser = models()
    prompt, target = question_prompt([1, 2, 3, 4]), mx.array([VOCAB - 1] * 4)
    loss_fn = lambda f: nn.losses.cross_entropy(cache_to_cache(receiver, sharer, f, prompt, training=True), target).mean()  # noqa: E731
    _, grads = nn.value_and_grad(fuser, loss_fn)(fuser)
    flat = dict(tree_flatten(grads))
    assert any(float(mx.abs(g).max()) > 0 for g in flat.values())
    assert all(np.isfinite(np.array(g)).all() for g in flat.values())
    # the base models are untouched by construction: nothing in `fuser` references them
    assert not any("blocks" in name or "embed" in name for name in flat)


def test_gumbel_gate_is_noisy_in_training_and_hard_at_inference():
    logit = mx.array(0.3)
    mx.random.seed(0)
    train = [float(gumbel_sigmoid(logit, 0.5, True)) for _ in range(20)]
    assert all(0 < t < 1 for t in train) and len(set(train)) > 10
    assert float(gumbel_sigmoid(logit, 0.5, False)) == 1.0
    assert float(gumbel_sigmoid(mx.array(-0.3), 0.5, False)) == 0.0


def test_temperature_anneals_geometrically():
    assert anneal(0, 100) == 1.0
    assert abs(anneal(100, 100) - 0.1) < 1e-9
    assert abs(anneal(50, 100) - (0.1**0.5)) < 1e-9
