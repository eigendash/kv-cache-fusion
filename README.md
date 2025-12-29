# kv-cache-fusion

A small MLX implementation of cache-to-cache communication between two language models: a trainable fuser reads the key/value cache of one model (the sharer), maps it into the cache space of another (the receiver), and adds it to the receiver's own cache, so the receiver can answer using knowledge it never learned.

The idea comes from *Cache-to-Cache: Direct Semantic Communication Between Large Language Models* (arXiv:2510.03215). This repository is an independent, toy-scale implementation written from the paper's description. It is not a reproduction of the paper's experiments or numbers, which use pretrained Qwen models on real benchmarks. Everything here runs on a laptop in a few minutes on Apple silicon.

## What is implemented

The fuser follows the structure described in the paper. For each receiver layer n, it takes the receiver's cache and the cache of a paired sharer layer G(n), concatenates them, and passes the result through a projection layer and a feature-fusion layer. A dynamic weighting step then rescales each attention head's contribution as a function of the input, and a learned per-layer gate decides whether the result is added at all. The output is a residual update to the receiver's cache, so the receiver's own information is kept. Keys and values have separate fusers. The gate is a Gumbel-sigmoid whose temperature is annealed from 1.0 to 0.1 during training and replaced by a hard threshold at inference. Both language models are frozen; only the fuser receives gradients, and the loss is the receiver's next-token loss on the answer, computed from the fused cache.

Layers of models with different depth are paired from the output end (the last receiver layer with the last sharer layer, and so on), which is the paper's terminal alignment. A depth-normalised pairing is included as an alternative.

## Setup

There are no pretrained models here, so the experiment builds a world in which communication has something to carry. There are 256 entities, each named by two syllable tokens and each carrying one of 24 attribute values. The sharer is trained on 128 of them (set A) and the receiver on a disjoint 128 (set B). The receiver is also trained to copy a value from a statement placed earlier in its context, which is what makes text-to-text communication possible. Neither model can answer questions about the other's entities from its own weights.

The sharer has 6 layers and width 96, the receiver 3 layers and width 64, both with 4 heads, so caches differ in depth and in width. Both use learned absolute position embeddings, so cached keys carry no rotary phase. They share a vocabulary, so the token-alignment step in the paper is not needed here. A question is the prompt `Q s1 s2 ?`. The receiver generates its answer from a single decoding step that starts with an `ANS` token and attends to the (fused) prompt cache. The sharer answers directly after the question mark, so the knowledge is retrieved while the prompt is being prefilled and is present in its cache.

Set A is split. The fuser trains on questions about 96 of the sharer's entities and is evaluated on the other 32, which it has never been asked about. A fuser that memorised entity-to-value pairs would fail there, so that column measures whether it transfers information from the sharer's cache. As a control, each C2C evaluation is repeated with the sharer's cache taken from a different question; a fuser that really reads the cache should drop to chance.

Both models were trained for 4000 steps (AdamW, learning rate 2e-3 with cosine decay, batch 128) and the fuser for 3000 steps with the same optimiser. The models were trained once; each fuser configuration was trained with three seeds. The full log is in `results/run_log.txt`.

## Results

Chance is 1/24, about 0.042. Baselines without a fuser:

| questions about | receiver alone | sharer alone | text-to-text |
|---|---:|---:|---:|
| sharer's facts, unseen by fuser | 0.062 | 1.000 | 1.000 |
| sharer's facts, seen by fuser | 0.021 | 1.000 | 1.000 |
| receiver's own facts | 1.000 | 0.023 | 0.023 |

Cache-to-cache accuracy, mean and standard deviation over three fuser seeds. The last column feeds the fuser a sharer cache computed for a different question.

| fuser trained on | questions about | C2C | mismatched sharer cache |
|---|---|---:|---:|
| sharer's questions only | sharer's facts, unseen | 0.552 ± 0.097 | 0.031 ± 0.026 |
| | sharer's facts, seen | 1.000 ± 0.000 | 0.090 ± 0.010 |
| | receiver's own facts | 0.029 ± 0.013 | 0.078 ± 0.006 |
| plus receiver's own questions | sharer's facts, unseen | 0.271 ± 0.053 | 0.010 ± 0.015 |
| | sharer's facts, seen | 1.000 ± 0.000 | 0.104 ± 0.009 |
| | receiver's own facts | 1.000 ± 0.000 | 0.279 ± 0.013 |
| same, depth-matched layers | sharer's facts, unseen | 0.208 ± 0.078 | 0.031 ± 0.026 |
| | sharer's facts, seen | 1.000 ± 0.000 | 0.080 ± 0.018 |
| | receiver's own facts | 1.000 ± 0.000 | 0.266 ± 0.033 |

## What the results do and do not show

The fuser does carry information across models. On questions the receiver cannot answer alone, accuracy rises from near chance to between 0.2 and 0.55 on entities the fuser never saw during training, and to 1.0 on entities it did see. Replacing the sharer's cache with one computed for a different question brings accuracy back to chance, so the fuser is reading the cache and not guessing from the receiver's own state.

The generalisation to unseen entities is partial. Perfect accuracy on seen entities next to 0.27 to 0.55 on unseen ones suggests the fuser has partly memorised entity-to-value pairs in its own parameters, which a fuser of this size could do; I did not test that directly. The numbers vary considerably between seeds, so the ordering of the configurations below should be treated as tentative.

Training the fuser only on questions the sharer can answer has a cost. Such a fuser transfers best to unseen entities (0.552) but overwrites the receiver's own knowledge: accuracy on the receiver's own facts falls from 1.000 to 0.029, and all three layers' gates stay open. Mixing in questions the receiver already knows removes that damage completely (1.000) but roughly halves the transfer to unseen entities (0.271). In this toy the two goals pull against each other, and I did not find a setting that achieves both. The mismatched-cache column on the receiver's own facts (0.279 for the mixed fuser) shows the fuser has not learned to ignore the sharer for those questions. With the matching cache the receiver stays correct, but a cache from another question drops it to 0.279, so it still depends on the sharer for those questions.

Cache-to-cache did not beat text-to-text here, and could not be expected to. When the sharer writes its answer as a single token, the text channel is lossless, so text-to-text matches the sharer's own accuracy (1.000). The paper's argument is that text discards semantic information that a cache preserves, and that argument needs knowledge that does not collapse to one token. This toy has no such case, so the experiment says nothing for or against that claim.

Terminal alignment (0.271) did slightly better than depth-matched alignment (0.208) on unseen entities, in line with the paper's choice, but the difference is smaller than the seed-to-seed spread and should not be read as evidence either way.

## Choices where the paper leaves room

The paper describes the dynamic weighting module as an input-aware head modulation layer without fixing its form. Here it is a linear map from the concatenated caches to one sigmoid weight per head and token. The projection is a two-layer perceptron with hidden width 128 and a GELU. The gate logit starts at zero, the output layer of the fusion step is scaled by 0.1 at initialisation, and the temperature follows a geometric schedule from 1.0 to 0.1. The paper's token-alignment procedure for models with different tokenizers is not implemented, because the two models share a vocabulary. The answer is a single token, so multi-token generation from a fused cache is untested.

## Limitations

The models are tiny, the task is a lookup table, and the sharer's knowledge is one token wide. The two language models were trained once, so variation from their initialisation is not measured; the three seeds vary only the fuser. There is no analysis of the fused cache itself, such as the effective-rank measurement in the paper, and no comparison against fine-tuning the receiver directly, which the paper uses to separate the value of the sharer from the value of extra trainable parameters. I make no wall-clock claims, since the runs shared the machine with other jobs.

## Running it

```
pip install -e ".[dev]"
pytest
python scripts/run_experiment.py --lm-steps 4000 --fuser-steps 3000 --seeds 3
```

The package is `kv_fusion`: `lm.py` is the transformer that exposes and accepts caches, `world.py` builds the facts and training batches, `fuser.py` holds the fuser and layer alignments, `comm.py` has the three ways of answering, and `training.py` the training loops. The tests (18) check that decoding on a cached prefix equals a single forward pass, that a closed gate reproduces the receiver's own output exactly, that gradients reach only the fuser, and that the data splits do not leak.
