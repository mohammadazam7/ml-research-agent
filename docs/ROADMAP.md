# Roadmap

Target: v1.0 on **Jan 9, 2027**. Each milestone ends with a pull request, a tagged release
and a changelog entry.

- [x] **M0 (Nov 9): foundation.** Repository, CI, design doc, roadmap.
- [ ] **M1 (Nov 13 to 24): runner and checks without a language model.** Dataset profiling,
  evaluation planning (random, grouped and time-based splits), leakage checks, baselines,
  a sandboxed experiment runner with tracking, calibration and slice reports. Fully
  deterministic and tested, so the agent builds on parts that are already trusted.
- [ ] **M2 (Nov 25 to Dec 8): the agent.** Model-provider interface (hosted APIs and any
  OpenAI-compatible local endpoint) with a scripted provider for tests, the plan, run,
  inspect and revise loop with a budget, and `mlagent run`.
- [ ] **M3 (Dec 9 to 22): the benchmark.** Tabular tasks with leakage traps, scoring
  against baselines, cost and time per task, `mlagent bench`, first published results.
- [ ] **M4 (Dec 23 to Jan 2): reports.** Model cards, an experiment comparison report and
  an interactive view of runs.
- [ ] **M5 (Jan 3 to 9): v1.0.** Package published, benchmark results for several models,
  demo video, final design doc and ADRs.
