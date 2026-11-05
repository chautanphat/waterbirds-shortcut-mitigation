#!/bin/bash

DEVICE="cuda:0"
RUNPATH="./ablations/auxmasks"

declare -a CONFIGS=(
    "ours_us undersample"
    "ours_us_op undersample_optimal"
    "ours_op optimal"
    "ours_os_op oversample_optimal"
    "ours_os oversample_optimal"
)

for SEED in 0 1 2; do
    for CONFIG in "${CONFIGS[@]}"; do
        read -r RUN_LABEL AUX_LABEL <<< "$CONFIG"

        for MODE in ours_detector ours_generate ours; do
            python run.py \
                --device="$DEVICE" \
                --results_dir="$RUNPATH" \
                --mode="$MODE" \
                --run_label="$RUN_LABEL" \
                --random_seed="$SEED" \
                --dataset_type="waterbird" \
                --detector_dataset="waterbirds_from_segmentation" \
                --waterbirds_use_minority=false \
                --aux_label_type="$AUX_LABEL"
        done

        rm -rf "./data/waterbird_complete95_forest2water2/abs_seed${SEED}"
        rm -rf "./methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed${SEED}"
    done
done
