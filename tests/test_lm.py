import mlx.core as mx
import numpy as np

from kv_fusion.lm import LMConfig, TinyLM


def model(layers=3, dim=32, heads=4):
    mx.random.seed(0)
    return TinyLM(LMConfig(vocab=20, dim=dim, heads=heads, layers=layers))


def test_prefix_decoding_matches_one_pass():
    # Running a suffix on top of a cached prefix must equal running the whole sequence.
    lm = model()
    tokens = mx.random.randint(0, 20, (3, 9))
    full, _ = lm(tokens)
    _, prefix = lm(tokens[:, :5])
    rest, _ = lm(tokens[:, 5:], prefix=prefix)
    np.testing.assert_allclose(np.array(rest), np.array(full[:, 5:]), atol=1e-5)


def test_cache_shapes_and_causality():
    lm = model(layers=2, dim=24, heads=3)
    a = mx.random.randint(0, 20, (2, 8))
    b = mx.concatenate([a[:, :5], mx.random.randint(0, 20, (2, 3))], axis=1)
    la, ca = lm(a)
    lb, _ = lm(b)
    assert len(ca) == 2 and ca[0][0].shape == (2, 3, 8, 8)
    np.testing.assert_allclose(np.array(la[:, :5]), np.array(lb[:, :5]), atol=1e-5)


def test_prefix_changes_the_continuation():
    lm = model()
    tokens = mx.random.randint(0, 20, (1, 4))
    _, p1 = lm(mx.array([[1, 2, 3]]))
    _, p2 = lm(mx.array([[4, 5, 6]]))
    l1, _ = lm(tokens, prefix=p1)
    l2, _ = lm(tokens, prefix=p2)
    assert not np.allclose(np.array(l1), np.array(l2), atol=1e-4)
