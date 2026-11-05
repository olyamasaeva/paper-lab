# June 2026 run, goal-aware judge (8 topics)

The authoritative result so far. It replaces the [pilot](2026-06-pilot.md), which ran
before the judge attributed gaps.

**Means:** score **7.44 / 10**, coverage **62%**. Coverage is understated by construction:
it counts all concepts of the reference, including the by-design ones.

| topic | score | coverage | real deficits / by design |
|---|---|---|---|
| NLP w11 interpretability | 7 | 60% | 4 / 1 |
| RecSys w04 two-tower | 7.5 | 61% | 5 / 3 |
| NLP w07 PEFT / LoRA | 7.5 | 85% | 2 / 4 |
| NLP w01 embeddings | 7.5 | 85% | 1 / 5 |
| NLP w04 transfer / BERT | 7 | 43% | 3 / 5 |
| NLP w06 prompting / ICL | 7.5 | 53% | 4 / 4 |
| NLP w08 efficiency / quantisation | 8 | 44% | 3 / 6 |
| RecSys w07 DCN-v2 ranking | 7.5 | 64% | 2 / 6 |

## Five recurring real deficits → the completeness contract

All five are domain-agnostic.

1. **[high, ×6]** The core of the method was moved into an Inline Question, a bonus or
   "next time" instead of a stub: the student finishes without building the core.
   → Contract point 1.
2. **[high, ×5]** The end-to-end run was not carried to the metric on the task: it stopped
   at a proxy (train loss, layer MSE, recall only) instead of downstream accuracy,
   perplexity or nDCG. → Contract point 2.
3. **[med, ×3]** "Why it works" stayed prose and was never materialised as a number or a
   derivation, although numerical diagnostics are the generator's strength. → Point 4.
4. **[med, ×2]** Applying the method (inference) was bypassed with a projection instead of
   a real run: steered generation not run, chain-of-thought not parsed. → Point 3.
5. **[med, ×2]** A family of variants was described in words, never put side by side on
   the same data and metric. → Point 5.

## By design (not deficits, correctly excluded)

Course engineering in the reference homework: HF/peft/TRL API, GPTQ on real hardware,
dataset ETL (BPE, temporal splits, collate masks), web-scale datasets (Yambda, IMDB,
wikitext), GPU scale, off-topic pipelines (RLHF/PPO in the LoRA homework, an MT
translator in the embeddings homework).

## Strengths (better than the reference in all 8; keep them)

Numerical `Expected ≈ …` self-checks with diagnostics; the core implemented from scratch;
Inline Questions and experiments first; LaTeX derivations; the genealogy of the method
and a mechanistic "why"; a debug area ordered from cheap to expensive checks; controlled
synthetic checks with a known ground-truth effect.

**Conclusion.** The gaps were not in style or diagnostics, where the generator is ahead of
the reference. They were in one thing: the method was not carried through to the end as a
method. One contract closes that, and the skill stays domain-agnostic.
