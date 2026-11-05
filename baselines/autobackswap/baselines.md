# Reproducing Results

All experiments use:

```bash
DEVICE="cuda:0"
```

## Waterbirds (Table 1)

Change to "false" to exclude minority samples from the training data:
```bash
WB_USE_MINORITY="true"
```

### ERM

```bash
python run.py \
--device=$DEVICE \
--dataset_type="waterbird" \
--waterbirds_use_minority=$WB_USE_MINORITY \
--mode="erm" \
--random_seed=0 \
--run_label="erm"
```

### ERM + Heavy Augmentations

```bash
python run.py \
--device=$DEVICE \
--dataset_type="waterbird" \
--waterbirds_use_minority=$WB_USE_MINORITY \
--mode="erm" \
--heavy_augmentations=true \
--random_seed=0 \
--run_label="erm_ha"
```

### DFR

```bash
python run.py \
--device=$DEVICE \
--dataset_type="waterbird" \
--waterbirds_use_minority=$WB_USE_MINORITY \
--mode="dfr" \
--random_seed=0 \
--run_label="dfr"
```

### AFR

```bash
python run.py \
--device=$DEVICE \
--dataset_type="waterbird" \
--waterbirds_use_minority=$WB_USE_MINORITY \
--mode="afr" \
--random_seed=0 \
--run_label="afr"
```

### GroupDRO

```bash
python run.py \
--device=$DEVICE \
--dataset_type="waterbird" \
--mode="group_dro" \
--waterbirds_use_minority=$WB_USE_MINORITY \
--random_seed=0 \
--run_label="group_dro"
```

### CORAL

```bash
python run.py \
--device=$DEVICE \
--dataset_type="waterbird" \
--mode="coral" \
--waterbirds_use_minority=$WB_USE_MINORITY \
--random_seed=0 \
--run_label="coral"
```

### Chang et al.

Generate counterfactual data once:

```bash
python run.py \
--device=$DEVICE \
--dataset_type="waterbird" \
--mode="chang_generate"
```

Train:

```bash
python run.py \
--device=$DEVICE \
--dataset_type="waterbird" \
--mode="chang" \
--waterbirds_use_minority=$WB_USE_MINORITY \
--random_seed=0 \
--run_label="chang"
```

### Foundation Models

SAM3:

```bash
python run.py \
--device=$DEVICE \
--dataset_type="waterbird" \
--mode="sam3" \
--random_seed=0 \
--run_label="sam3"
```

QWEN3:

```bash
CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
--dataset_type="waterbirds" \
--mode="qwen3" \
--random_seed=0 \
--run_label="qwen3"
```

CLIP:

```bash
python run.py \
--device=$DEVICE \
--dataset_type="waterbird" \
--mode="clip" \
--random_seed=0 \
--run_label="clip"
```

---

## Spawrious (Table 2)

You can choose which Spawrious variant to use:
```bash
DEVICE="cuda:0"
VARIANT="o2o_easy"
```

Available variants:

```text
o2o_easy
o2o_medium
o2o_hard
m2m_easy
m2m_medium
m2m_hard
```

### ERM

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spawrious/${VARIANT}" \
--mode="erm" \
--random_seed=0 \
--run_label="erm"
```

### ERM + Heavy Augmentations

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spawrious/${VARIANT}" \
--mode="erm" \
--heavy_augmentations=true \
--random_seed=0 \
--run_label="erm_ha"
```

### DFR

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spawrious/${VARIANT}" \
--mode="dfr" \
--random_seed=0 \
--run_label="dfr"
```

### AFR

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spawrious/${VARIANT}" \
--mode="afr" \
--random_seed=0 \
--run_label="afr"
```

### GroupDRO

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spawrious/${VARIANT}" \
--mode="group_dro" \
--random_seed=0 \
--run_label="group_dro"
```

### CORAL

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spawrious/${VARIANT}" \
--mode="coral" \
--random_seed=0 \
--run_label="coral"
```

### Chang et al.

Generate counterfactual data once:

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spawrious/${VARIANT}" \
--mode="chang_generate"
```

Train:

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spawrious/${VARIANT}" \
--mode="chang" \
--random_seed=0 \
--run_label="chang"
```

### Foundation Models

SAM3:

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spawrious/${VARIANT}" \
--mode="sam3" \
--random_seed=0 \
--run_label="sam3"
```

QWEN3:

```bash
CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
--dataset_type="spawrious/${VARIANT}" \
--mode="qwen3" \
--random_seed=0 \
--run_label="qwen3"
```

CLIP:

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spawrious/${VARIANT}" \
--mode="clip" \
--random_seed=0 \
--run_label="clip"
```

---
## Spurious Vehicles Experiments (Table 3)

```bash
DEVICE="cuda:0"
```

### ERM

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spurious_vehicles_m2m" \
--mode="erm" \
--random_seed=0 \
--run_label="erm"
```

### ERM + Heavy Augmentations

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spurious_vehicles_m2m" \
--mode="erm" \
--heavy_augmentations=true \
--random_seed=0 \
--run_label="erm_ha"
```

### DFR

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spurious_vehicles_m2m" \
--mode="dfr" \
--random_seed=0 \
--run_label="dfr"
```

### AFR

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spurious_vehicles_m2m" \
--mode="afr" \
--random_seed=0 \
--run_label="afr"
```

### GroupDRO

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spurious_vehicles_m2m" \
--mode="group_dro" \
--random_seed=0 \
--run_label="group_dro"
```

### CORAL

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spurious_vehicles_m2m" \
--mode="coral" \
--random_seed=0 \
--run_label="coral"
```

### Chang et al.

Generate counterfactual data once:

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spurious_vehicles_m2m" \
--mode="chang_generate"
```

Train:

```bash
python run.py \
--device=$DEVICE \
--dataset_type="spurious_vehicles_m2m" \
--mode="chang" \
--random_seed=0 \
--run_label="chang"
```
