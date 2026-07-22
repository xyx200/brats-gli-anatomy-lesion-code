import argparse
import glob
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import nibabel as nib
import numpy as np


LESION_CLASS_INDEX = 4
DEFAULT_EXCLUDE_CONFIG = {
    "0845": [0, 1, 3],
    "1138": [1],
    "1149": [0, 1, 2],
    "1158": [0, 2],
    "1182": [0, 1, 3],
}


class SimpleEntropyFusion:
    def __init__(self, epsilon=1e-8, ignore_background=False):
        self.epsilon = epsilon
        self.ignore_background = ignore_background

    def _calculate_voxel_wise_entropy(self, probability_map):
        target_probs = probability_map[1:, ...] if self.ignore_background else probability_map
        probs = np.clip(target_probs, self.epsilon, 1.0)
        return -np.sum(probs * np.log(probs), axis=0)

    def fuse(self, model_outputs):
        if not model_outputs:
            raise ValueError("No modality probabilities remain for fusion.")

        stacked_probs = np.stack(model_outputs, axis=0)
        entropy_stack = np.stack(
            [self._calculate_voxel_wise_entropy(probability_map) for probability_map in model_outputs],
            axis=0,
        )
        raw_weights = np.exp(-entropy_stack)
        weights = raw_weights / (np.sum(raw_weights, axis=0, keepdims=True) + self.epsilon)
        fused_prob = np.sum(weights[:, np.newaxis, ...] * stacked_probs, axis=0)
        return fused_prob, np.argmax(fused_prob, axis=0)


def load_npz_prob(path):
    try:
        with np.load(path) as data:
            for key in ("softmax", "arr_0", "prob", "vol_data"):
                if key in data:
                    return data[key]
            return data[list(data.keys())[0]]
    except Exception as exc:
        raise IOError(f"Unable to read NPZ file {path}: {exc}") from exc


def apply_expert_constraints(prob_map, expert_mask, lesion_class_index=LESION_CLASS_INDEX):
    modified_prob = prob_map.copy()
    lesion_mask_area = expert_mask > 0
    modified_prob[:, lesion_mask_area] = 0.0
    modified_prob[lesion_class_index, lesion_mask_area] = 1.0
    background_area = expert_mask == 0
    modified_prob[lesion_class_index, background_area] = 0.0
    current_sums = np.sum(modified_prob, axis=0)
    safe_sums = current_sums.copy()
    safe_sums[safe_sums == 0] = 1.0
    modified_prob[:, background_area] /= safe_sums[background_area]
    return modified_prob


def process_single_case(
    expert_path,
    npz_root,
    output_root,
    exclude_config,
    lesion_class_index=LESION_CLASS_INDEX,
):
    filename = os.path.basename(expert_path)
    case_id = "unknown"
    try:
        base_name = filename.replace(".nii.gz", "")
        case_id = base_name.split("-")[0]

        nii_obj = nib.load(expert_path)
        expert_data_original = nii_obj.get_fdata().astype(np.uint8)
        expert_data = np.transpose(expert_data_original, (2, 1, 0))
        skip_list = exclude_config.get(case_id, [])

        probability_maps = []
        for modality_index in range(4):
            if modality_index in skip_list:
                continue

            npz_name = f"{case_id}_{modality_index:04d}_aggregated_probs.npz"
            npz_path = os.path.join(npz_root, npz_name)
            if not os.path.exists(npz_path):
                return f"[Error] ID {case_id}: missing file {npz_name}"

            probability_map = load_npz_prob(npz_path)
            if probability_map.shape[1:] != expert_data.shape:
                return f"[Error] ID {case_id}: shape mismatch for {npz_name}"

            probability_maps.append(
                apply_expert_constraints(probability_map, expert_data, lesion_class_index)
            )

        if not probability_maps:
            return f"[Warning] ID {case_id}: all modalities were excluded; fusion skipped."

        _, fused_label = SimpleEntropyFusion(ignore_background=False).fuse(probability_maps)
        fused_label = np.transpose(fused_label, (2, 1, 0)).astype(np.uint8)
        header = nii_obj.header.copy()
        header.set_data_dtype(np.uint8)
        save_path = os.path.join(output_root, f"{case_id}.nii.gz")
        nib.save(nib.Nifti1Image(fused_label, nii_obj.affine, header=header), save_path)
        return f"[Complete] {case_id}: fused {len(probability_maps)} modalities."
    except Exception as exc:
        return f"[Error] ID {case_id}: {exc}"


def build_parser():
    parser = argparse.ArgumentParser(description="Fuse aggregated healthy-tissue maps with lesion masks.")
    parser.add_argument("--npz-root", required=True, help="Folder containing aggregated probability .npz files.")
    parser.add_argument("--expert-root", required=True, help="Folder containing lesion-mask NIfTI files.")
    parser.add_argument("--output-dir", required=True, help="Folder for fused joint-label NIfTI files.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.isdir(args.npz_root):
        raise FileNotFoundError(f"Aggregated probability directory does not exist: {args.npz_root}")
    if not os.path.isdir(args.expert_root):
        raise FileNotFoundError(f"Lesion-mask directory does not exist: {args.expert_root}")
    os.makedirs(args.output_dir, exist_ok=True)

    pattern = re.compile(r"^\d{4}(-seg)?\.nii\.gz$")
    expert_files = sorted(
        path
        for path in glob.glob(os.path.join(args.expert_root, "*.nii.gz"))
        if pattern.match(os.path.basename(path))
    )

    print(f"Aggregated probability directory: {args.npz_root}")
    print(f"Lesion-mask directory: {args.expert_root}")
    print(f"Cases: {len(expert_files)}")
    print(f"Configured modality exclusions: {len(DEFAULT_EXCLUDE_CONFIG)} cases")

    process_func = partial(
        process_single_case,
        npz_root=args.npz_root,
        output_root=args.output_dir,
        exclude_config=DEFAULT_EXCLUDE_CONFIG,
    )
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(process_func, expert_files), 1):
            if "[Error]" in result or "[Warning]" in result or index % 50 == 0:
                print(f"[{index}/{len(expert_files)}] {result}")
    print(f"Fusion completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
