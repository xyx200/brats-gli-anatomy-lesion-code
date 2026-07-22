# Evaluation Integration

This directory contains evaluation code adapted from the MedNeXt v1 package. It provides the joint-label metric behavior reported for this resource. To use these files, obtain and install MedNeXt from its official repository, then replace the corresponding MedNeXt evaluation files with `evaluator.py` and
`metrics.py` from this directory. This replacement is solely for reproducing the evaluation convention reported for this resource.

For `Dice`/`DSC`, when both prediction and reference are empty, this release reports `1.0`.

## Adapted Metric Rules

For `Hausdorff Distance 95`, this release applies the following explicit rule:

- both prediction and reference empty: `0.0`;
- only one is empty, or either mask is completely full: `374.0`;
- otherwise: compute MedPy HD95 and cap the returned value at `374.0`;
- a MedPy calculation failure also returns `374.0`.

These Dice/DSC and HD95 rules must be reported with any results produced by the adapted evaluator.

## License

`evaluator.py` and `metrics.py` retain the DKFZ Apache License 2.0 header from the upstream MedNeXt/nnUNet evaluation implementation and carry a notice of the modifications made for this resource. They are distributed under Apache License 2.0, not solely under the MIT license for project-authored code. See `../LICENSE-APACHE-2.0.txt` and `../THIRD_PARTY_NOTICES.md`.

The upstream MedNeXt project is available at: https://github.com/MIC-DKFZ/MedNeXt. This release does not modify or redistribute the MedNeXt model architecture, trainer, or model weights.
