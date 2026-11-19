# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- Baselines scored with the plan's metric: a prior that ignores the features and a
  ridge model on the usable columns, with every fill value and scale taken from the
  training rows. ROC AUC, average precision, macro F1 and MAE are implemented here and
  agree with scikit-learn to 1e-15 on random scores with ties.
- Project foundation: package layout, CI, contribution and security guides, design doc
  and roadmap.
