# BraTS-GLI-Anatomy-Lesion Processing Code

**Version:** v1.0.0

This repository releases the project-authored processing scripts and the
evaluation adaptation used with the BraTS-GLI-Anatomy-Lesion derivative
resource. It is distributed separately from the data package.

## Released Code Scope

This repository provides:

1. A wrapper for externally supplied healthy-tissue probability-map inference.
2. Aggregation of source probabilities into the eight-class joint label space.
3. Entropy-weighted fusion with a supplied lesion-mask constraint.
4. Lesion inpainting orchestration with an externally installed NiftySeg,
   intended for cases marked `image_repair_required=repair_required`.
5. Evaluation files adapted from the MedNeXt/nnUNet evaluation package for the
   reported joint-label metric behavior.

This repository does **not** distribute BraTS imaging data, model weights,
MedNeXt model or training code, DeepWMH, LST-AI, TumorSynth, NiftySeg, or a
complete third-party software environment. MedNeXt was used as a validation
model; this release does not modify or redistribute its model architecture,
trainer, or weights.

## Repository Structure

```text
CODE/
|-- .gitignore
|-- LICENSE.txt
|-- LICENSE-APACHE-2.0.txt
|-- README.md
|-- THIRD_PARTY_NOTICES.md
|-- requirements.txt
|-- run_inpainting.bash
|-- evaluation/
|   |-- README.md
|   |-- evaluator.py
|   `-- metrics.py
`-- healthy label/
    |-- nnunet.sh
    |-- aggregate_seg.py
    `-- shang.py
```

## External Software And Models

External tools and trained models must be obtained from their official sources
and are not redistributed here:

- MedNeXt: https://github.com/MIC-DKFZ/MedNeXt
- NiftySeg: https://github.com/KCL-BMEIS/NiftySeg
- DeepWMH: https://github.com/lchdl/DeepWMH
- LST-AI: https://github.com/CompImg/LST-AI
- TumorSynth: https://github.com/fprados/TumorSynth

The Python packages imported by the released scripts are listed in
`requirements.txt`. Installation and licensing of external tools and models
remain governed by their official sources.

## Released Script Interfaces

### 1. Healthy-Tissue Probability-Map Inference Wrapper

`healthy label/nnunet.sh` invokes an externally available `nnUNet_predict`
command with externally supplied trained-model assets. Inputs must be named
`<subject>_0000.nii.gz` through `<subject>_0003.nii.gz`.

```bash
export NNUNET_RAW_DATA_BASE=/path/to/nnUNet_raw
export NNUNET_PREPROCESSED=/path/to/nnUNet_preprocessed
export NNUNET_RESULTS_FOLDER=/path/to/nnUNet_trained_models
export NNUNET_INPUT_DIR=/path/to/imagesTr
export NNUNET_OUTPUT_DIR=/path/to/inference_outputs
bash "healthy label/nnunet.sh"
```

Optional variables are `MODEL_NAME`, `TASK`, `MAX_JOBS`, and `START_INDEX`.
This wrapper does not include a trained model, training configuration, or
model-weight download.

### 2. Class Aggregation

`healthy label/aggregate_seg.py` aggregates externally generated probability
maps into the joint label space:

```bash
python "healthy label/aggregate_seg.py" \
  --input-dir /path/to/inference_outputs \
  --output-dir /path/to/aggregated_outputs \
  --workers 16
```

| Label | Region |
| --- | --- |
| 0 | Background |
| 1 | Cortical gray matter |
| 2 | Basal ganglia |
| 3 | White matter |
| 4 | Lesion |
| 5 | Ventricles |
| 6 | Cerebellum |
| 7 | Brainstem |

The script saves aggregated probability maps, NIfTI segmentations, and
`voxel_stats_summary.csv`. The source-channel mapping encoded by this script
must be used with the corresponding external source probability model.

### 3. Entropy-Weighted Fusion

`healthy label/shang.py` fuses aggregated per-modality probabilities and
imposes the supplied lesion mask as class `4`:

```bash
python "healthy label/shang.py" \
  --npz-root /path/to/aggregated_outputs \
  --expert-root /path/to/lesion_masks \
  --output-dir /path/to/fused_outputs \
  --workers 8
```

The current implementation accepts lesion-mask filenames matching
`NNNN.nii.gz` or `NNNN-seg.nii.gz`, where `NNNN` is a four-digit case ID. It
encodes modality exclusions for cases `0845`, `1138`, `1149`, `1158`, and
`1182`; changes to those exclusions alter this release's curation decision and
must be reported.

The associated Synapse release provides `metadata/id_mapping.csv`, which maps
these four-digit IDs back to the original BraTS-compatible case identifiers and
file names for locating excluded TumorSynth-derived modality probability maps.

### 4. Lesion Inpainting

`run_inpainting.bash` orchestrates NiftySeg executables obtained separately
from the official project. The script uses the pre-joint-label convention in
which label `2` is inpainted and a dilated label `1` region is excluded from
the brain mask. For cases marked `image_repair_required=repair_required` in the
associated data resource, use the image-repair labels from
`repair_labels/modified/` as the inpainting-mask input when reconstructing the
repaired MRI inputs used during curation. Expert-negative cases do not use
these repair labels.

```bash
export BASE_DATA_DIR=/path/to/t1n_t1c_t2w_t2f_inputs
export LABEL_DIR=/path/to/inpainting_masks
export OUTPUT_DIR=/path/to/inpainted_outputs
bash run_inpainting.bash
```

Optional variables are `NIFTYSEG_BIN_DIR`, `MAX_JOBS`, `DILATION`, and
`SEARCH_SIZE`.

### 5. Joint-Label Evaluation Adaptation

`evaluation/evaluator.py` and `evaluation/metrics.py` are Apache-2.0-licensed
adaptations of files from the MedNeXt/nnUNet evaluation package. To reproduce
the joint-label evaluation used in this work, obtain and install MedNeXt from
its official repository, then replace the corresponding MedNeXt evaluation
files with the two adapted files released in `evaluation/`.

The adapted DSC and HD95 metric rules are documented in
[`evaluation/README.md`](evaluation/README.md). The released evaluation files
do not constitute a modification of the MedNeXt model architecture, trainer,
or model weights.

## Licenses And Attribution

Project-authored scripts and documentation are released under the
**MIT License** in [`LICENSE.txt`](LICENSE.txt).

`evaluation/evaluator.py` and `evaluation/metrics.py` are modified from the
MedNeXt/nnUNet evaluation package and are distributed under the
**Apache License, Version 2.0**, provided in
[`LICENSE-APACHE-2.0.txt`](LICENSE-APACHE-2.0.txt). Their upstream attribution
and modification notice are described in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The associated derivative data resource is licensed separately from this code
package and remains subject to applicable upstream BraTS requirements. The
data resource is hosted on Synapse at
https://www.synapse.org/Synapse:syn75210889/wiki/.

## Associated Data Resource

The associated BraTS-GLI-Anatomy-Lesion v1.0.0 data resource is a labels-only Synapse resource with a public landing page and an access-controlled versioned label release:

- Public Synapse landing page and Wiki: https://www.synapse.org/Synapse:syn75210889/wiki/640990
- Synapse project ID: `syn75210889`
- Controlled label release: `syn75260521`
- Upstream BraTS 2023 project ID: `syn51156910`
- Upstream BraTS access page: https://www.synapse.org/Synapse:syn51156910/wiki/627000
- Code repository: https://github.com/xyx200/brats-gli-anatomy-lesion-code
- Official contact: xxy200200@stu.xjtu.edu.cn
- Citation: Cite the BraTS-GLI-Anatomy-Lesion resource article using its final bibliographic citation when available. The associated data resource may additionally be cited alongside the resource article using its DOI: https://doi.org/10.7303/SYN75210889.

The data package contains labels for 1,251 BraTS 2023-GLI cases, split into 394 purified cases and 857 extended cases. Its labels-only release manifest lists 1,367 NIfTI label files: 1,251 released unified segmentation labels and 116 image-repair labels. The image-repair labels are derived from the SVID/WMH label resources associated with: Multi-Disease Segmentation of Gliomas and White Matter Hyperintensities in the BraTS Data Using a 3D Convolutional Neural Network, and the source repository `johncolby/svid_paper`.

MRI images are not redistributed in this Synapse project. Users must obtain the corresponding BraTS 2023-GLI MRI data directly from upstream Synapse ID `syn51156910`; for the 116 cases marked `image_repair_required=repair_required`, users should additionally apply the image-repair labels with the provided lesion-inpainting script (`run_inpainting.bash`) to reconstruct the repaired MRI inputs used during curation. Expert-negative cases are marked `image_repair_required=no_repair_required` and do not have repair labels.

The controlled release includes the following metadata files: `participants.csv`, `labels_description.json`, `metadata/cases.csv`, `metadata/id_mapping.csv`, `metadata/release_manifest.csv`, `metadata/data_dictionary.md`, and `scripts/download.sh`.

The Synapse project landing page and Wiki are publicly viewable. The versioned label release (`syn75260521`) remains access-controlled. Access requires a registered Synapse account and approved access to the upstream BraTS 2023 data through `syn51156910`. Follow the email-based request instructions on the public Wiki; after manual approval, the maintainer grants the requester's Synapse account read and download access to `syn75260521`. The labels are released under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). This data license does not remove or replace the John Colby MIT notice for the SVID/WMH repair-label source material and does not override applicable upstream BraTS data-use terms.

Downloaded label files and image-repair labels must not be shared with anyone who has not been approved for access to the derivative label resource.
