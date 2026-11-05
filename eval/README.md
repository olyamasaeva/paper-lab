# Evaluating paper-lab against real course homework

How do you judge a generated assignment? There is no ground truth for "a good homework".
The closest thing is homework written by people who teach the subject for a living. This
eval uses the public courses of the Yandex School of Data Analysis
([yandexdataschool](https://github.com/yandexdataschool)) as that reference.

## Protocol

1. **Pairs.** For each topic, the course repo has a lecture (PDF, HTML or slides) and a
   homework notebook. The generator sees **only the lecture**. The homework is held out
   as the reference.
2. **Generate.** paper-lab builds an assignment from the lecture.
3. **Judge.** An LLM judge compares the generated notebook with the reference homework
   using [`rubric.md`](rubric.md): a 0–10 score, concept coverage, and a list of gaps.
4. **Attribute every gap.** Each gap is labelled `real_deficit` (the generator missed
   something the method needs) or `by_design_or_domain` (course engineering the generator
   deliberately leaves out: HF/peft API drills, GPTQ on real hardware, dataset ETL,
   web-scale data, off-topic pipelines). Without this split, coverage punishes the
   generator for skipping things it should skip.
5. **Fix the generator, not the notebook.** Recurring real deficits become rules in
   `SKILL.md` (the completeness contract). The skill stays domain-agnostic: no rule may
   mention a specific topic.

The topic list is in [`golden_set.csv`](golden_set.csv): 16 positive pairs with a
fetchable lecture, 2 groups of partial pairs (video lectures only), and 5 negative
controls where no source is available. The manifest links to the public course repos and
copies no course content.

## Results so far

Run of June 2026 on 8 of the 16 positive topics, with the goal-aware judge
([details](results/2026-06-goal-aware.md)).

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
| **mean** | **7.44** | **62%** | |

Coverage is computed against **all** concepts of the reference, including the
by-design ones, so it understates what the generator covers on purpose.

**What the judge found.** Five deficits recurred across topics, none of them about style
or diagnostics:

1. The core of the method (the paper's contribution) was pushed into an Inline Question
   or a bonus instead of a stub, so the student could finish without building it (high, seen 6 times).
2. The end-to-end run stopped at a proxy (train loss, layer MSE, recall only) instead of
   the metric on the task (high, 5 times).
3. "Why it works" stayed prose, not a number or a derivation (medium, 3 times).
4. Applying the method (steered generation, parsing chain-of-thought) was projected,
   not actually run (medium, 2 times).
5. A family of variants was described in words, never compared on the same data and
   metric (medium, 2 times).

Each one became a point of the completeness contract in `SKILL.md`.

**Where the generated notebooks beat the reference in all 8 topics:** numerical
`Expected ≈ …` self-checks with diagnostics; the core implemented from scratch; Inline
Questions and "experiments first"; derivations; controlled synthetic checks with a known
ground-truth effect.

## Limitations, honestly

- **n = 8.** Eight topics from two courses, mostly NLP. Not enough to claim much beyond
  "the recurring deficits are real and the contract targets them".
- **One judge, no human agreement yet.** The judge is an LLM. There is no second human
  rater and no κ yet, so the scores are a signal, not a measurement.
- **The contract has not been re-scored.** The fixes went into the skill after this run;
  a rerun is needed to show they help.
- **The negative controls have not been run.** They check that the generator does not
  make up a course when there is no source.
- **Ingest was fragile.** In the earlier pilot, 3 of 5 runs failed because whole PDFs were
  read into the context. `scripts/extract_source.py` fixes that.

## Next

- [ ] Eval harness as code in this folder (fetch pairs → generate → judge → aggregate),
      reproducible from `golden_set.csv`.
- [ ] Rerun on all 16 positive topics with the completeness contract in place.
- [ ] Run the 5 negative controls.
- [ ] A second human rater on a subset, agreement with the judge (κ), Wilson intervals.
