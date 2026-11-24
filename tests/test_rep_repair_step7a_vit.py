"""Unit tests for Direction 2 · Phase C · Step 7a ViT-B/16 model, patching, and LN-affine controls."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from src.models.vit import (
    VIT_BLOCK_SITES,
    ViTClassifier,
    configure_vit_ln_params,
    pool_vit_fg_tokens,
    vit_foreground_token_drift,
)


class Step7aViTTests(unittest.TestCase):
    def test_extract_block_tokens_and_forward_from_site_exact_match(self):
        torch.manual_seed(0)
        model = ViTClassifier(weights=None)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.inference_mode():
            logits = model(x)
            feats = model.extract_features(x)
            toks = model.extract_block_tokens(x)

        self.assertLess(float((logits - toks["logits"]).abs().max()), 1e-5)
        self.assertLess(float((feats - toks["pooled"]).abs().max()), 1e-5)
        self.assertEqual(tuple(toks["patch_map"].shape), (2, 768, 14, 14))
        for site in ("embed", "block_0", "block_5", "block_11"):
            self.assertEqual(tuple(toks[site].shape), (2, 197, 768))
            with torch.inference_mode():
                res = model.forward_from_site(toks[site], site)
            self.assertLess(float((res["logits"] - logits).abs().max()), 1e-5)
            self.assertLess(float((res["pooled"] - feats).abs().max()), 1e-5)

    def test_configure_vit_ln_params_freezes_all_attention_mlp_and_embeddings(self):
        torch.manual_seed(1)
        model = ViTClassifier(weights=None)
        params_late = configure_vit_ln_params(model, "ln_late")
        params_all = configure_vit_ln_params(model, "ln_all")
        self.assertGreater(len(params_all), len(params_late))

        # Re-check ln_late exact scope: blocks 0..7 LN frozen, blocks 8..11 + final LN trainable, all Linear/Conv frozen except head.
        configure_vit_ln_params(model, "ln_late")
        self.assertFalse(model.vit.conv_proj.weight.requires_grad)
        self.assertFalse(model.vit.class_token.requires_grad)
        self.assertFalse(model.vit.encoder.pos_embedding.requires_grad)
        for mod in model.vit.encoder.layers[0].modules():
            if isinstance(mod, nn.LayerNorm):
                self.assertFalse(mod.weight.requires_grad)
        for mod in model.vit.encoder.layers[11].modules():
            if isinstance(mod, nn.LayerNorm):
                self.assertTrue(mod.weight.requires_grad)
            elif isinstance(mod, nn.Linear):
                self.assertFalse(mod.weight.requires_grad)
        self.assertTrue(model.vit.encoder.ln.weight.requires_grad)
        self.assertTrue(model.vit.heads.head.weight.requires_grad)

    def test_foreground_token_drift_and_pool_fg_tokens(self):
        torch.manual_seed(2)
        model = ViTClassifier(weights=None)
        model.eval()
        x1 = torch.randn(1, 3, 224, 224)
        x2 = torch.randn(1, 3, 224, 224)
        mask = np.zeros((224, 224), dtype=np.float32)
        mask[64:160, 64:160] = 1.0
        with torch.inference_mode():
            t1 = model.extract_block_tokens(x1)
            t2 = model.extract_block_tokens(x2)
        d_same = vit_foreground_token_drift(t1, t1, mask)
        d_diff = vit_foreground_token_drift(t1, t2, mask)
        for site in (*VIT_BLOCK_SITES, "final_cls"):
            self.assertAlmostEqual(d_same[site], 0.0, places=6)
            self.assertGreater(d_diff[site], 0.0)

        # Oracle FG pool must recover exact foreground value when foreground vs background are distinct constants.
        pmap = np.zeros((1, 4, 14, 14), dtype=np.float32)
        pmap[:, :, 5:9, 5:9] = 3.5
        m14 = np.zeros((1, 14, 14), dtype=np.float32)
        m14[:, 5:9, 5:9] = 1.0
        pooled = pool_vit_fg_tokens(pmap, m14)
        self.assertTrue(np.allclose(pooled, 3.5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
