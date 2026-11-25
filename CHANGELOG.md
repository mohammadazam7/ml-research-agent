# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- Experiment runner: each script runs in a fresh directory with `python -I`, an
  environment holding only `PATH` and the data path, and wall-clock, CPU and memory
  limits; every run goes to an append-only JSON-lines log with hashes of the script
  and data, its status, time and metrics.
- Calibration and slice reports: reliability bins, expected calibration error and
  Brier score; the metric per value of chosen columns, flagging slices of 30 or more
  rows that are more than 0.05 worse than overall.
- Baselines scored with the plan's metric: a prior that ignores the features and a
  ridge model on the usable columns, with every fill value and scale taken from the
  training rows. ROC AUC, average precision, macro F1 and MAE are implemented here and
  agree with scikit-learn to 1e-15 on random scores with ties.
- Project foundation: package layout, CI, contribution and security guides, design doc
  and roadmap.
