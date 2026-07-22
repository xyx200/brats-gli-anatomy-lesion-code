#!/usr/bin/env bash

set -euo pipefail
export PYTHONWARNINGS="ignore"

: "${NNUNET_RAW_DATA_BASE:?Set NNUNET_RAW_DATA_BASE to the nnUNet raw-data root.}"
: "${NNUNET_PREPROCESSED:?Set NNUNET_PREPROCESSED to the nnUNet preprocessed root.}"
: "${NNUNET_RESULTS_FOLDER:?Set NNUNET_RESULTS_FOLDER to the trained-model root.}"
: "${NNUNET_INPUT_DIR:?Set NNUNET_INPUT_DIR to the folder containing modality images.}"
: "${NNUNET_OUTPUT_DIR:?Set NNUNET_OUTPUT_DIR to the output folder.}"

export nnUNet_raw_data_base="$NNUNET_RAW_DATA_BASE"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"
export RESULTS_FOLDER="$NNUNET_RESULTS_FOLDER"

INPUT_DIR="$NNUNET_INPUT_DIR"
OUTPUT_DIR="$NNUNET_OUTPUT_DIR"
MODEL_NAME="${MODEL_NAME:-nnUNetPlansv2.1}"
TASK="${TASK:-002}"
MAX_JOBS="${MAX_JOBS:-3}"
START_INDEX="${START_INDEX:-1}"

mkdir -p "$OUTPUT_DIR"
shopt -s nullglob

process_subject() {
    local subject="$1"
    local working_dir
    local count=0

    echo "Processing subject: ${subject}"
    working_dir="$(mktemp -d -t TumorSynth_XXXXXX)"

    for mod in 0000 0001 0002 0003; do
        local current_file="${INPUT_DIR}/${subject}_${mod}.nii.gz"
        local final_nii="${OUTPUT_DIR}/${subject}_${mod}.nii.gz"

        if [[ ! -f "$current_file" ]]; then
            echo "[Warning] Modality ${mod} not found for ${subject}."
            continue
        fi

        local mod_dir="${working_dir}/mod_${mod}"
        local out_dir="${working_dir}/out_${mod}"
        mkdir -p "$mod_dir" "$out_dir"
        cp "$current_file" "${mod_dir}/input_0000.nii.gz"

        echo "  Inference on modality: ${mod}"
        nnUNet_predict \
            -i "$mod_dir" \
            -o "$out_dir" \
            -tr nnUNetTrainerV2 \
            -ctr nnUNetTrainerV2CascadeFullRes \
            -m 3d_fullres \
            -p "$MODEL_NAME" \
            -t "$TASK" \
            -f 0 1 2 3 4 \
            --save_npz >/dev/null

        local nii_candidates=("$out_dir"/*.nii.gz)
        local npz_candidates=("$out_dir"/*.npz)
        local pkl_candidates=("$out_dir"/*.pkl)

        if ((${#nii_candidates[@]} == 0)); then
            echo "[Error] Prediction failed for ${subject}, modality ${mod}." >&2
            continue
        fi

        mv "${nii_candidates[0]}" "$final_nii"
        if ((${#npz_candidates[@]} > 0)); then
            mv "${npz_candidates[0]}" "${OUTPUT_DIR}/${subject}_${mod}.npz"
        fi
        if ((${#pkl_candidates[@]} > 0)); then
            mv "${pkl_candidates[0]}" "${OUTPUT_DIR}/${subject}_${mod}.pkl"
        fi
        count=$((count + 1))
    done

    rm -rf "$working_dir"
    echo "[Complete] ${subject}: saved ${count} modality results."
}

mapfile -t subjects < <(
    find "$INPUT_DIR" -maxdepth 1 -type f -name '*.nii.gz' -printf '%f\n' |
        sed 's/_[0-9]\{4\}\.nii\.gz$//' |
        sort -u
)

total_subjects="${#subjects[@]}"
echo "Found ${total_subjects} subjects; starting at index ${START_INDEX} with ${MAX_JOBS} parallel jobs."

current_idx=0
for subject in "${subjects[@]}"; do
    current_idx=$((current_idx + 1))
    if [[ "$current_idx" -lt "$START_INDEX" ]]; then
        echo "[${current_idx}/${total_subjects}] Skipping ${subject} before start index."
        continue
    fi

    echo "[${current_idx}/${total_subjects}] Dispatching ${subject}."
    process_subject "$subject" &
    if [[ "$(jobs -r -p | wc -l)" -ge "$MAX_JOBS" ]]; then
        wait -n
    fi
done

wait
echo "Inference completed."
