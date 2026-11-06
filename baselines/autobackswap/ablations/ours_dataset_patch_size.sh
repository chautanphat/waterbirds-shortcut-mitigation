#!/bin/bash

DEVICE="cuda:0"
RUNPATH="./ablations/dspr"

# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds50_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=50
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds50_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=50
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds50_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=50
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds100_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=100
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds100_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=100
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds100_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=100
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds200_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=200
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds200_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=200
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds200_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=200
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds400_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=400
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds400_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=400
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds400_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=400
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds800_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=800
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds800_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=800
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds800_pr4" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=4 --detector_n_samples=800
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0


# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds50_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=50
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds50_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=50
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds50_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=50
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds100_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=100
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds100_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=100
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds100_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=100
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds200_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=200
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds200_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=200
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds200_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=200
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds400_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=400
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds400_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=400
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds400_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=400
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds800_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=800
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds800_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=800
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds800_pr8" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=8 --detector_n_samples=800
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0


# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds50_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=50
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds50_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=50
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds50_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=50
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds100_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=100
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds100_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=100
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds100_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=100
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds200_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=200
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds200_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=200
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds200_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=200
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds400_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=400
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds400_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=400
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds400_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=400
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds800_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=800
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds800_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=800
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds800_pr16" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=16 --detector_n_samples=800
# rm -R data/waterbird_complete95_forest2water2/abs_seed0
# rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0


# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds50_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=50
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds50_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=50
# python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds50_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=50
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds100_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=100
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds100_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=100
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds100_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=100
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds200_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=200
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds200_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=200
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds200_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=200
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds400_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=400
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds400_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=400
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds400_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=400
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds800_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=800
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds800_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=800
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds800_pr28" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=28 --detector_n_samples=800
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0


python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds50_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=50
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds50_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=50
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds50_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=50
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds100_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=100
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds100_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=100
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds100_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=100
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds200_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=200
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds200_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=200
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds200_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=200
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds400_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=400
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds400_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=400
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds400_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=400
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds800_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=800
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds800_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=800
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds800_pr56" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=56 --detector_n_samples=800
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0


python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds50_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=50
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds50_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=50
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds50_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=50
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds100_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=100
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds100_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=100
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds100_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=100
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds200_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=200
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds200_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=200
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds200_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=200
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds400_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=400
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds400_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=400
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds400_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=400
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_detector" --run_label="ours_ds800_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=800
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours_generate" --run_label="ours_ds800_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=800
python run.py --device=$DEVICE --results_dir=$RUNPATH --mode="ours" --run_label="ours_ds800_pr112" --random_seed=0 --dataset_type="waterbird" --detector_dataset="waterbirds_from_segmentation" --waterbirds_use_minority=false --detector_patch_resolution=112 --detector_n_samples=800
rm -R data/waterbird_complete95_forest2water2/abs_seed0
rm -R methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0