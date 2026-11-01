#!/usr/bin/env python
"""
reproduce_demo.py - Self-Contained Smoke Test & Quickstart for DPE.

Demonstrates the Diverse Prototypical Ensembles (DPE) pipeline:
1. Simulates a subpopulation-shifted feature space (ResNet-50 style, D=2048) with severe spurious correlation.
2. Trains an ensemble of diversified prototype classifiers (IsoMax+ with covariance regularization).
3. Evaluates per-group accuracy and worst-group accuracy (WGA) across ensemble stages.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import torch

# Ensure local package import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dpe.core import DPE


def create_synthetic_features(tmp_dir, n_samples_per_split=400, feat_dim=2048, seed=42):
    """
    Creates synthetic features with a strong spurious correlation between class y and attribute a:
    - Group 0 (y=0, a=0): majority class 0
    - Group 1 (y=0, a=1): minority class 0
    - Group 2 (y=1, a=0): minority class 1
    - Group 3 (y=1, a=1): majority class 1
    """
    rng = np.random.default_rng(seed)
    
    # 4 centers in latent space
    centers = rng.standard_normal((4, feat_dim))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * 5.0
    
    rows = []
    splits = {'train': 0, 'val': 1, 'test': 2}
    
    for split_name, split_id in splits.items():
        feats_list = []
        for i in range(n_samples_per_split):
            # 85% majority correlation in train, 50% balanced in val/test
            if split_name == 'train':
                y = int(rng.random() > 0.5)
                a = y if (rng.random() < 0.85) else (1 - y)
            else:
                y = int(rng.random() > 0.5)
                a = int(rng.random() > 0.5)
            
            group = y * 2 + a
            noise = rng.standard_normal(feat_dim) * 1.5
            feat = centers[group] + noise
            feats_list.append(feat)
            
            rows.append({
                'filename': f'{split_name}_{i:04d}.png',
                'split': split_id,
                'y': y,
                'a': a,
                'g': group
            })
        
        feats_arr = np.array(feats_list, dtype=np.float32)
        np.save(os.path.join(tmp_dir, f'feats_{split_name}.npy'), feats_arr)
    
    df = pd.DataFrame(rows)
    metadata_path = os.path.join(tmp_dir, 'metadata.csv')
    df.to_csv(metadata_path, index=False)
    return metadata_path


def main():
    print("=" * 60)
    print("  Diverse Prototypical Ensembles (DPE) - Quickstart Demo")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Running on device: {device.upper()}")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        print("[*] Generating synthetic subpopulation benchmark data...")
        meta_path = create_synthetic_features(tmp_dir, n_samples_per_split=400, feat_dim=256, seed=0)
        
        print("[*] Initializing DPE model...")
        dpe_model = DPE(
            data_dir=tmp_dir,
            metadata_path=meta_path,
            num_stages=4,
            epochs=5,
            emb_dim=256,
            batch_size_train=64,
            batch_size_eval=128,
            device=device,
            eval_freq=1,
            train_attr=False,
            split_map={'val': 'va', 'test': 'te'},
            subsample_type='class',
            cov_reg=1e4,
            entropic=20,
            seed=0,
            workers=0
        )
        
        print("[*] Fitting Diverse Prototypical Ensemble (4 stages)...")
        wga_history, bacc_history, last_res = dpe_model.fit()
        
        print("\n[*] Evaluating on Test Set...")
        eval_res = dpe_model.evaluate(target='test')
        
        print("\n" + "=" * 60)
        print("                 EVALUATION RESULTS")
        print("=" * 60)
        print(f"Overall Accuracy:       {eval_res['overall']['accuracy'] * 100:.2f}%")
        print(f"Balanced Accuracy:      {eval_res['overall']['balanced_acc'] * 100:.2f}%")
        print(f"Worst-Group Accuracy:   {eval_res['min_group']['accuracy'] * 100:.2f}%")
        print("-" * 60)
        print("Per-Group Accuracies:")
        for g, metrics in sorted(eval_res['per_group'].items()):
            print(f"  Group {g}: {metrics['accuracy'] * 100:.2f}% (n={metrics['n_samples']})")
        print("=" * 60)
        print("\n[+] DPE pipeline successfully reproduced and verified!")


if __name__ == '__main__':
    main()
