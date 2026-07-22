# Third-Party Notices

## MedNeXt / nnUNet Evaluation Code

`evaluation/evaluator.py` and `evaluation/metrics.py` are adapted from the MedNeXt/nnUNet evaluation package at `nnunet_mednext/evaluation`, distributed by the Division of Medical Image Computing, German Cancer Research Center (DKFZ). The upstream source is: https://github.com/MIC-DKFZ/MedNeXt

```text
Copyright 2020 Division of Medical Image Computing,
German Cancer Research Center (DKFZ), Heidelberg, Germany
Licensed under the Apache License, Version 2.0.
```

The adapted files retain the upstream copyright and Apache License 2.0 headers. They are distributed under the Apache License, Version 2.0, whose
full text is included in `LICENSE-APACHE-2.0.txt`; they are not covered solely by the repository's MIT license.

The BraTS-GLI-Anatomy-Lesion project modified these files in 2026 for its joint-label evaluation reporting, including explicit HD95 handling for empty or full masks, failed calculations, and the maximum reported distance. For Dice, when both prediction and reference are empty, this release reports 1.0. The project did not modify or redistribute the MedNeXt model architecture, training implementation, or model weights.

The MIT license in `LICENSE.txt` applies to project-authored scripts and documentation in this release. It does not remove obligations associated with third-party Apache-2.0-licensed files.
