#!/bin/bash

DEVICE="cuda:0"
RUNPATH="./ablations/infilling"

INFILLING_TYPES=(
  "gan"
  "noise"
  "gray"
  "average"
  "scramble_pixel"
  "scramble_patch"
)

SEEDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do

  echo "========= Seed $SEED ========="

  python run.py \
    --device=$DEVICE \
    --dataset_type="waterbird" \
    --mode="ours_detector" \
    --detector_dataset="waterbirds_from_segmentation" \
    --random_seed="$SEED"

  python run.py \
    --device=$DEVICE \
    --dataset_type="waterbird" \
    --mode="ours_generate" \
    --detector_dataset="waterbirds_from_segmentation" \
    --random_seed="$SEED"

  for INFILLING in "${INFILLING_TYPES[@]}"; do
    echo "--- $INFILLING ---"
    python run.py \
      --mode="ours" \
      --device="$DEVICE" \
      --dataset_type="waterbird" \
      --random_seed="$SEED" \
      --results_dir="$RUNPATH" \
      --infilling="$INFILLING" \
      --detector_dataset="waterbirds_from_segmentation" \
      --run_label="ours_${INFILLING}"

  done

  rm -rf "./data/waterbird_complete95_forest2water2/abs_seed${SEED}"
  rm -rf "./methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed${SEED}"

done
