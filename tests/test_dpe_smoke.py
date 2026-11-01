import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

# Add baselines/dpe4subpop to sys.path
BASELINES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'baselines', 'dpe4subpop')
sys.path.insert(0, BASELINES_DIR)

from src.dpe.isomaxplus import IsoMaxPlusLossFirstPart, IsoMaxPlusLossSecondPart
from src.dpe.core import DPE


class TestDPESmoke(unittest.TestCase):

    def test_isomaxplus_forward(self):
        num_features = 64
        num_classes = 2
        head = IsoMaxPlusLossFirstPart(num_features, num_classes)
        loss_fn = IsoMaxPlusLossSecondPart(entropic_scale=10.0)

        x = torch.randn(8, num_features)
        y = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1])

        logits = head(x)
        self.assertEqual(logits.shape, (8, num_classes))
        self.assertFalse(torch.isnan(logits).any())

        loss = loss_fn(logits, y)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0.0)

    def test_dpe_smoke_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            n_samples = 60
            dim = 32
            splits = {'val': 1, 'test': 2}
            rows = []
            rng = np.random.default_rng(42)

            for s_name, s_id in splits.items():
                feats = rng.standard_normal((n_samples, dim)).astype(np.float32)
                np.save(os.path.join(tmp_dir, f'feats_{s_name}.npy'), feats)
                for i in range(n_samples):
                    y = int(i % 2)
                    a = int((i // 2) % 2)
                    rows.append({
                        'filename': f'{s_name}_{i}.png',
                        'split': s_id,
                        'y': y,
                        'a': a,
                        'g': y * 2 + a
                    })

            meta_df = pd.DataFrame(rows)
            meta_path = os.path.join(tmp_dir, 'metadata.csv')
            meta_df.to_csv(meta_path, index=False)

            dpe = DPE(
                data_dir=tmp_dir,
                metadata_path=meta_path,
                num_stages=2,
                epochs=2,
                emb_dim=dim,
                batch_size_train=16,
                batch_size_eval=32,
                device='cpu',
                eval_freq=1,
                train_attr=False,
                split_map={'val': 'va', 'test': 'te'},
                subsample_type='class',
                cov_reg=1e3,
                entropic=10,
                seed=0,
                workers=0
            )

            wga_hist, bacc_hist, last_res = dpe.fit()
            self.assertEqual(len(wga_hist), 2)
            self.assertEqual(len(dpe.ensemble), 2)

            eval_res = dpe.evaluate(target='test')
            self.assertIn('overall', eval_res)
            self.assertIn('min_group', eval_res)
            self.assertIn('accuracy', eval_res['overall'])
            self.assertIn('accuracy', eval_res['min_group'])


if __name__ == '__main__':
    unittest.main()
