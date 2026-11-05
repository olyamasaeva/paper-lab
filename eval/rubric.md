# Judge rubric

The judge gets three inputs: the **lecture** the generator saw, the **reference homework**
written by the course staff, and the **generated notebook**. It returns JSON.

The rubric was used in the June 2026 run. It is written down here from that run so the
harness can be rebuilt from it; the harness code itself is not in the repo yet.

## Instructions to the judge

You are reviewing a generated assignment against a reference homework on the same
lecture. The goal of the generated assignment is to make a student **understand the
method**: implement its core, run it end to end, measure it with the metric of the task.
The reference is course homework; it also trains engineering skills that the generator
deliberately leaves out.

1. **List the concepts of the reference.** Every method, step, metric or experiment the
   reference homework makes the student do. One line each.
2. **Mark coverage.** For each concept: `covered` (the generated notebook makes the
   student do it), `mentioned` (only in prose or an Inline Question), `missing`.
3. **Attribute every non-covered concept.**
   - `real_deficit`: the concept belongs to the method, and a student who finishes the
     generated notebook has not built or measured it. Examples: the paper's contribution
     sits in a bonus; the run stops at train loss when the method is measured by nDCG;
     a family of variants is described but never compared.
   - `by_design_or_domain`: course engineering outside the goal: library API drills
     (HF, peft, TRL), running GPTQ on real hardware, dataset ETL (BPE training, temporal
     splits, collate masks), web-scale datasets, GPU-scale training, off-topic pipelines.
   When unsure, prefer `real_deficit` and say why.
4. **Severity** of each real deficit: `high` (the core of the method or the metric on the
   task is missing), `med` (the explanation or comparison is weaker), `low`.
5. **Strengths.** What the generated notebook does better than the reference, concretely.
6. **Score 0–10** for how well the generated notebook reaches its goal (understanding the
   method through implementing and measuring it). Not for how closely it copies the
   reference.

## Output

```json
{
  "topic": "nlp w07 peft",
  "score": 7.5,
  "concepts": [
    {"name": "LoRA forward pass", "status": "covered"},
    {"name": "peft library config", "status": "missing", "attribution": "by_design_or_domain"},
    {"name": "downstream accuracy after fine-tuning", "status": "mentioned",
     "attribution": "real_deficit", "severity": "high",
     "why": "the notebook stops at train loss; LoRA is judged by task accuracy"}
  ],
  "coverage": 0.85,
  "strengths": ["rank sweep with a numerical check of the parameter count", "..."],
  "summary": "two sentences"
}
```

`coverage` = covered concepts / all concepts of the reference (by-design ones included).

## Aggregation

- Mean score and mean coverage across topics.
- Real deficits grouped across topics by what went wrong, ignoring the topic: a deficit
  that recurs in 2+ topics becomes a candidate rule for `SKILL.md`. A rule must not name
  a topic.
- For the negative controls (no lecture available) the expected behaviour is a refusal or
  an explicit "no source" header, not an assignment. Count fabrications.
