import argparse
import glob
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import nibabel as nib
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm

    USE_TQDM = True
except ImportError:
    USE_TQDM = False


def get_label_mapping_lut(max_id=255):
    """Create the lookup table that maps source labels to classes 0 through 7."""
    lut = np.ones(max_id + 1, dtype=np.int16)
    source_groups = {
        0: [0, 12],
        2: [7, 8, 9, 10, 14, 15, 16],
        3: [1],
        4: [18],
        5: [3, 4, 11],
        6: [5, 6],
        7: [13, 17],
    }
    for target_class, source_labels in source_groups.items():
        for source_label in source_labels:
            if source_label <= max_id:
                lut[source_label] = target_class
    return lut


MAPPING_LUT = get_label_mapping_lut()


def aggregate_probabilities(original_softmax):
    """Aggregate source softmax channels into the eight target channels."""
    n_channels, depth, height, width = original_softmax.shape
    new_softmax = np.zeros((8, depth, height, width), dtype=np.float32)

    for source_class in range(n_channels):
        target_class = MAPPING_LUT[source_class] if source_class < len(MAPPING_LUT) else 1
        new_softmax[target_class] += original_softmax[source_class]

    return new_softmax


def process_single_file(input_path, output_dir):
    filename = os.path.basename(input_path)
    nii_path = os.path.join(os.path.dirname(input_path), filename.replace(".npz", ".nii.gz"))

    try:
        affine = nib.load(nii_path).affine if os.path.exists(nii_path) else np.eye(4)
        with np.load(input_path) as data:
            softmax = data["softmax"] if "softmax" in data else data[list(data.keys())[0]]

        new_softmax = aggregate_probabilities(softmax)
        new_seg = np.argmax(new_softmax, axis=0).astype(np.int16)
        # Neural-network output is (Z, Y, X); NIfTI output is stored as (X, Y, Z).
        new_seg = np.transpose(new_seg, (2, 1, 0))

        if "_softmax.npz" in filename:
            npz_name = filename.replace("_softmax.npz", "_aggregated_probs.npz")
            nii_name = filename.replace("_softmax.npz", "_aggregated_seg.nii.gz")
        else:
            npz_name = filename.replace(".npz", "_aggregated_probs.npz")
            nii_name = filename.replace(".npz", "_aggregated_seg.nii.gz")

        np.savez_compressed(os.path.join(output_dir, npz_name), softmax=new_softmax, affine=affine)
        nib.save(nib.Nifti1Image(new_seg, affine), os.path.join(output_dir, nii_name))

        return {"filename": filename, "tumor_voxels": int(np.sum(new_seg == 4))}
    except Exception as exc:
        pid = multiprocessing.current_process().pid
        print(f"[Error-PID:{pid}] Failed to process {filename}: {exc}")
        return None


def calculate_voxel_stats(data_dir, target_labels=(1, 2, 3, 4, 5, 6, 7)):
    """Calculate foreground label voxel counts for aggregated NIfTI files."""
    files = glob.glob(os.path.join(data_dir, "*_aggregated_seg.nii.gz"))
    if not files:
        print(f"Warning: no *_aggregated_seg.nii.gz files found under {data_dir}.")
        return None

    results = []
    print(f"Calculating voxel statistics for {len(files)} files.")
    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            flat_data = nib.load(file_path).get_fdata().astype(int).ravel()
            total_foreground = int(np.sum(flat_data > 0))
            counts = np.bincount(flat_data)
            row_data = {"Filename": file_name, "Total_Foreground": total_foreground}
            for label in target_labels:
                label_count = int(counts[label]) if label < len(counts) else 0
                ratio = label_count / total_foreground if total_foreground else 0.0
                row_data[f"Label_{label}_Count"] = label_count
                row_data[f"Label_{label}_Ratio"] = ratio
            results.append(row_data)
        except Exception as exc:
            print(f"Failed to calculate statistics for {file_name}: {exc}")
    return pd.DataFrame(results)


def build_parser():
    parser = argparse.ArgumentParser(description="Aggregate healthy-tissue probability maps.")
    parser.add_argument("--input-dir", required=True, help="Folder containing nnUNet probability .npz files.")
    parser.add_argument("--output-dir", required=True, help="Folder for aggregated probability and label files.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    all_npz = glob.glob(os.path.join(args.input_dir, "*.npz"))
    npz_files = [path for path in all_npz if "aggregated" not in os.path.basename(path)]
    total_files = len(npz_files)
    print(f"Found {total_files} input files in {args.input_dir}.")
    print(f"Output directory: {args.output_dir}")
    if not total_files:
        return

    process_func = partial(process_single_file, output_dir=args.output_dir)
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        iterator = executor.map(process_func, npz_files)
        results = list(tqdm(iterator, total=total_files, unit="file")) if USE_TQDM else list(iterator)

    successful = sum(result is not None for result in results)
    print(f"Aggregation completed in {time.time() - start_time:.2f} seconds: {successful}/{total_files} files.")
    dataframe = calculate_voxel_stats(args.output_dir)
    if dataframe is not None:
        output_csv = os.path.join(args.output_dir, "voxel_stats_summary.csv")
        dataframe.to_csv(output_csv, index=False)
        print(f"Saved statistics CSV: {output_csv}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

