# Design: ml-research-agent

**Author:** Mohammad Azam · **Status:** Draft · **Last updated:** Nov 9, 2026

## Context

Much of applied machine learning is routine and error-prone: profiling the data, choosing
a split, building features, training a baseline and a few stronger models, comparing them
fairly and writing it up. Language-model agents can write that code quickly, and they
repeat the classic mistakes just as quickly: random splits on time-ordered data, features
that encode the label, test sets reused for tuning, and reports that quote the best number
without its context.

## Goals

1. Run a tabular supervised learning project from a dataset and a goal to a model card,
   without a person writing code.
2. Make evaluation mistakes hard: choose and justify the split before training, detect
   leakage, keep a test set the agent evaluates on once.
3. Report honestly: baselines, uncertainty, calibration, error slices and what failed.
4. Run generated code safely: no network, bounded time and memory, confined files.
5. Measure the agent on a public benchmark, per model and per release.

## Non-goals

- Deep learning on images, audio or text in v1.0; tabular data only.
- Deployment and serving; the output is a trained model, a card and the code to retrain.
- Hyperparameter search at scale; a small budget of well-chosen experiments.

## Design

### The deterministic core (M1)

Everything that judges a model is plain, tested code, not model output:

| Component | Responsibility |
|---|---|
| Profiler | Column types, missingness, cardinality, target balance, candidate time and group columns. |
| Evaluation planner | Picks the split from the profile (time-based when a time column orders the outcome, grouped when rows share an entity) and fixes the metric and the held-out test set. |
| Leakage checks | Single features that predict the target almost perfectly, features observed after the prediction time, identical rows across train and test, target encodings fitted on the full data. |
| Baselines | Majority class or mean, and a regularised linear model. |
| Runner | Executes a training script in a separate process without network access, with time and memory limits, and records code, parameters, data hash, seed and metrics. |
| Reports | Calibration curves and Brier score, metrics by slice, and the experiment table. |

### The agent (M2)

The agent proposes features and models and writes training scripts; the core runs and
judges them. Its loop is plan, run, inspect results and checks, revise, within a budget of
experiments and cost. It can never mark its own result as valid: only the core's checks can.

### Model providers

The same small interface as in data-analyst-agent: hosted APIs and OpenAI-compatible local
endpoints, plus a scripted provider for deterministic tests.

## Alternatives considered

- **Classic AutoML only.** Strong at search, but does not reason about the split or leakage
  and does not explain itself. Included in the benchmark as a comparison.
- **Letting the agent judge its own results.** Simpler, and the main way agents produce
  confident nonsense. Rejected: judgement stays in tested code.
- **Running generated code in a container per experiment.** Stronger isolation but heavier
  to set up locally; a separate process with resource limits and no network is the default,
  with containers as an option later.

## Security and privacy

See [SECURITY.md](../SECURITY.md). Datasets may contain personal data; reports contain
aggregates only, and run directories stay local unless the user exports them.

## Testing plan

- Unit tests for every check, each with a dataset that must trigger it and one that must not.
- Sandbox tests: a script that tries to reach the network, read outside its directory or
  exceed its limits must fail.
- The agent loop tested end to end with the scripted provider.
- The benchmark run on a small sample in CI and in full per release.

## Open questions

- Which local model can write useful training code on a laptop CPU for the demo.
- How to score "caught the leakage trap" without rewarding agents that drop every feature.
