#!/usr/bin/env bash

set -euo pipefail

# Install NiftySeg before running this script:
# https://github.com/KCL-BMEIS/NiftySeg
# The installation must provide seg_FillLesions and seg_maths on PATH.
if [[ -n "${NIFTYSEG_BIN_DIR:-}" ]]; then
    export PATH="${NIFTYSEG_BIN_DIR}:$PATH"
fi

: "${BASE_DATA_DIR:?Set BASE_DATA_DIR to the folder containing -t1n, -t1c, -t2w and -t2f files.}"
: "${LABEL_DIR:?Set LABEL_DIR to the folder containing inpainting masks.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the output folder.}"

MAX_JOBS="${MAX_JOBS:-10}"
DILATION="${DILATION:-2}"
SEARCH_SIZE="${SEARCH_SIZE:-4}"

if ! command -v seg_FillLesions >/dev/null 2>&1 || ! command -v seg_maths >/dev/null 2>&1; then
    echo "Error: NiftySeg is required. Install it first and expose seg_FillLesions and seg_maths on PATH." >&2
    exit 1
fi

if [[ ! -d "$BASE_DATA_DIR" ]]; then
    echo "Error: input directory does not exist: $BASE_DATA_DIR" >&2
    exit 1
fi
if [[ ! -d "$LABEL_DIR" ]]; then
    echo "Error: label directory does not exist: $LABEL_DIR" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

process_subject() {
    local t1c_path="$1"
    local output_folder="$2"
    local filename
    local id
    local input_dir

    filename="$(basename "$t1c_path")"
    id="${filename%-t1c.nii.gz}"
    input_dir="$(dirname "$t1c_path")"

    local t1n_path="${input_dir}/${id}-t1n.nii.gz"
    local t2w_path="${input_dir}/${id}-t2w.nii.gz"
    local t2f_path="${input_dir}/${id}-t2f.nii.gz"
    local label_path="${LABEL_DIR}/${id}.nii.gz"
    if [[ ! -f "$label_path" && -f "${LABEL_DIR}/${id}-seg.nii.gz" ]]; then
        label_path="${LABEL_DIR}/${id}-seg.nii.gz"
    fi

    local t1n_out="${output_folder}/${id}-t1n.nii.gz"
    local t1c_out="${output_folder}/${id}-t1c.nii.gz"
    local t2w_out="${output_folder}/${id}-t2w.nii.gz"
    local t2f_out="${output_folder}/${id}-t2f.nii.gz"
    local brain_mask_temp="${output_folder}/${id}_brain_mask_final.nii.gz"
    local label1_dil_temp="${output_folder}/${id}_label1_dilated.nii.gz"
    local lesion_mask_temp="${output_folder}/${id}_lesion_label2_temp.nii.gz"

    if [[ ! -f "$t1n_path" || ! -f "$t2w_path" || ! -f "$t2f_path" ]]; then
        echo "[Skip] $id is missing one or more required modality files."
        return
    fi

    if [[ ! -f "$label_path" ]]; then
        echo "[Warning] $id has no label file; copying source images."
        cp "$t1n_path" "$t1n_out"
        cp "$t1c_path" "$t1c_out"
        cp "$t2w_path" "$t2w_out"
        cp "$t2f_path" "$t2f_out"
        return
    fi

    echo "[Processing] $id: creating masks and filling lesions."

    # Build a brain mask while excluding a dilated label-1 area.
    seg_maths "$label_path" -thr 0.7 -uthr 1.3 -bin -dil 3 "$label1_dil_temp"
    seg_maths "$t1n_path" -thr 0.0001 -bin -sub "$label1_dil_temp" -thr 0 -bin "$brain_mask_temp"

    # Extract label 2 as the lesion mask to be filled.
    seg_maths "$label_path" -thr 1.7 -uthr 2.3 -bin "$lesion_mask_temp"

    seg_FillLesions \
        -i "$t1n_path" \
        -l "$lesion_mask_temp" \
        -o "$t1n_out" \
        -mask "$brain_mask_temp" \
        -dil "$DILATION" \
        -size "$SEARCH_SIZE" \
        -v >/dev/null

    seg_FillLesions \
        -i "$t1c_path" \
        -l "$lesion_mask_temp" \
        -o "$t1c_out" \
        -mask "$brain_mask_temp" \
        -dil "$DILATION" \
        -size "$SEARCH_SIZE" \
        -v >/dev/null

    seg_FillLesions \
        -i "$t2f_path" \
        -l "$lesion_mask_temp" \
        -o "$t2f_out" \
        -mask "$brain_mask_temp" \
        -dil "$DILATION" \
        -size "$SEARCH_SIZE" \
        -v >/dev/null

    seg_FillLesions \
        -i "$t2w_path" \
        -l "$lesion_mask_temp" \
        -o "$t2w_out" \
        -mask "$brain_mask_temp" \
        -dil "$DILATION" \
        -size "$SEARCH_SIZE" \
        -v >/dev/null

    rm -f "$brain_mask_temp" "$lesion_mask_temp" "$label1_dil_temp"
    echo "[Complete] $id"
}

process_folder_parallel() {
    local input_folder="$1"
    local output_folder="$2"
    local count

    echo "Scanning: $input_folder"
    count="$(find "$input_folder" -maxdepth 1 -type f -name '*-t1c.nii.gz' | wc -l)"
    echo "Cases: $count; parallel jobs: $MAX_JOBS"

    for t1c_path in "$input_folder"/*-t1c.nii.gz; do
        [[ -e "$t1c_path" ]] || continue
        process_subject "$t1c_path" "$output_folder" &
        while [[ "$(jobs -r -p | wc -l)" -ge "$MAX_JOBS" ]]; do
            sleep 0.5
        done
    done
}

process_folder_parallel "$BASE_DATA_DIR" "$OUTPUT_DIR"
wait
echo "All cases completed."
