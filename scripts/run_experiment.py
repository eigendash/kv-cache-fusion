"""Train a sharer, a receiver and a fuser, then compare ways of answering.

    python scripts/run_experiment.py
    python scripts/run_experiment.py --lm-steps 4000 --fuser-steps 3000 --seeds 3
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kv_fusion import (  # noqa: E402
    CacheFuser,
    LMConfig,
    TinyLM,
    World,
    cache_to_cache,
    depth_layer_map,
    receiver_alone,
    sharer_alone,
    terminal_layer_map,
    text_to_text,
)
from kv_fusion.comm import accuracy  # noqa: E402
from kv_fusion.training import train_fuser, train_lm  # noqa: E402
from kv_fusion.world import VOCAB, question_targets  # noqa: E402


def evaluate(world, receiver, sharer, fuser):
    rng = np.random.default_rng(123)
    out = {}
    for name, ents in [("A_test", world.fuser_test), ("A_train", world.fuser_train), ("B", world.receiver)]:
        prompt, target = question_targets(world, ents)
        shuffled = question_targets(world, ents[rng.permutation(len(ents))])[0]
        row = {
            "receiver alone": accuracy(receiver_alone(receiver, prompt), target),
            "sharer alone": accuracy(sharer_alone(sharer, prompt), target),
            "text-to-text": accuracy(text_to_text(receiver, sharer, ents, prompt), target),
            "C2C": accuracy(cache_to_cache(receiver, sharer, fuser, prompt), target),
            "C2C, mismatched sharer cache": accuracy(
                cache_to_cache(receiver, sharer, fuser, prompt, sharer_prompt=shuffled), target
            ),
        }
        out[name] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-steps", type=int, default=4000)
    ap.add_argument("--fuser-steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3, help="fuser seeds (the two LMs are trained once)")
    ap.add_argument("--receiver-layers", type=int, default=3)
    ap.add_argument("--sharer-layers", type=int, default=6)
    args = ap.parse_args()

    world = World.make(seed=0)
    r_cfg = LMConfig(vocab=VOCAB, dim=64, heads=4, layers=args.receiver_layers)
    s_cfg = LMConfig(vocab=VOCAB, dim=96, heads=4, layers=args.sharer_layers)

    mx.random.seed(0)
    print("training sharer (knows set A)")
    sharer = train_lm(TinyLM(s_cfg), world, "sharer", args.lm_steps, log=print)
    print("training receiver (knows set B, can read context)")
    receiver = train_lm(TinyLM(r_cfg), world, "receiver", args.lm_steps, seed=1, log=print)
    for m in (sharer, receiver):
        m.freeze()

    configs = [
        ("terminal, sharer questions only", terminal_layer_map, 0.0),
        ("terminal, plus receiver's own questions", terminal_layer_map, 0.5),
        ("depth-matched, plus receiver's own questions", depth_layer_map, 0.5),
    ]
    results = {label: [] for label, _, _ in configs}
    for label, mapper, own in configs:
        for seed in range(args.seeds):
            mx.random.seed(100 + seed)
            fuser = CacheFuser(r_cfg, s_cfg, layer_map=mapper)
            print(f"training fuser ({label}, seed {seed})")
            train_fuser(fuser, receiver, sharer, world, args.fuser_steps, seed=seed, receiver_fraction=own, log=print)
            res = evaluate(world, receiver, sharer, fuser)
            res["gates"] = fuser.gates()
            results[label].append(res)

    def mean(label, split, key):
        v = [r[split][key] for r in results[label]]
        return np.mean(v), np.std(v)

    chance = 1 / 24
    print(f"\nchance is {chance:.3f}; fuser rows are mean +- std over {args.seeds} fuser seeds\n")
    names = {"A_test": "sharer's facts, unseen by fuser", "A_train": "sharer's facts, seen by fuser", "B": "receiver's own facts"}
    first = results[configs[0][0]][0]
    print("baselines (no fuser):\n")
    print("| questions about | receiver alone | sharer alone | text-to-text |")
    print("|---|---:|---:|---:|")
    for split, label in names.items():
        b = first[split]
        print(f"| {label} | {b['receiver alone']:.3f} | {b['sharer alone']:.3f} | {b['text-to-text']:.3f} |")
    print("\ncache-to-cache:\n")
    print("| fuser trained with | questions about | C2C | C2C with mismatched sharer cache |")
    print("|---|---|---:|---:|")
    for label, _, _ in configs:
        for split, name in names.items():
            c, mm = mean(label, split, "C2C"), mean(label, split, "C2C, mismatched sharer cache")
            print(f"| {label} | {name} | {c[0]:.3f} +- {c[1]:.3f} | {mm[0]:.3f} +- {mm[1]:.3f} |")
    for label, _, _ in configs:
        print(f"open gates per layer (keys, values), {label}, seed 0:", results[label][0]["gates"])


if __name__ == "__main__":
    main()
