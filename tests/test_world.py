import numpy as np

from kv_fusion.world import ANS, FACT, N_VALUES, PAD, QMARK, QUESTION, VALUE0, VOCAB, World, lm_batch, question_prompt


def test_splits_are_disjoint_and_cover_set_a():
    w = World.make(seed=1)
    assert not set(w.sharer) & set(w.receiver)
    assert not set(w.fuser_train) & set(w.fuser_test)
    assert set(w.fuser_train) | set(w.fuser_test) == set(w.sharer)


def test_question_prompt_layout():
    p = np.array(question_prompt([0, 17]))
    assert p.shape == (2, 4)
    assert (p[:, 0] == QUESTION).all() and (p[:, 3] == QMARK).all()


def test_supervised_positions_always_point_at_the_right_value():
    w = World.make(seed=2)
    rng = np.random.default_rng(0)
    for role in ("sharer", "receiver"):
        tokens, mask = lm_batch(w, role, rng, 200)
        tokens, mask = np.array(tokens), np.array(mask)
        assert (mask.sum(1) == 1).all()
        for row, m in zip(tokens, mask):
            target = row[1:][m.astype(bool)][0]
            assert VALUE0 <= target < VOCAB
            assert (row < VOCAB).all()


def test_receiver_context_examples_copy_the_context_value():
    w = World.make(seed=3)
    rng = np.random.default_rng(1)
    tokens, mask = lm_batch(w, "receiver", rng, 400)
    seen = 0
    for row, m in zip(np.array(tokens), np.array(mask)):
        if row[0] == FACT and row[4] == QUESTION:  # in-context example
            seen += 1
            assert row[8] == ANS and row[9] == row[3]
            assert m[8] == 1 and m[2] == 0  # the random statement value is not supervised
    assert seen > 100


def test_models_are_never_trained_on_the_other_models_facts():
    w = World.make(seed=4)
    rng = np.random.default_rng(2)
    from kv_fusion.world import SYLLABLE0, N_SYLLABLES

    tokens, mask = lm_batch(w, "sharer", rng, 300)
    for row in np.array(tokens):
        ent = (row[1] - SYLLABLE0) * N_SYLLABLES + (row[2] - SYLLABLE0)
        assert ent in set(w.sharer)
    assert N_VALUES == 24 and PAD == 0
