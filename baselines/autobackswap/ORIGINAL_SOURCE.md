# Original source

- **Method:** AutoBackSwap
- **Paper:** *Automated Background Swapping for Robustness against Spurious Backgrounds*, https://arxiv.org/abs/2606.32018
- **Repository:** https://github.com/cesro7/spur-corr
- **Commit:** `b060532c2632039ffc2918828780d2b696763af6`
- **Date imported:** 2026-11-05
- **How it was copied:** `git archive` of that commit (tracked files only). No `.git` folder, no checkpoints, no images. The upstream `README.md` is stored as `UPSTREAM_README.md`.
- **Submodule not copied:** `AOT-GAN-for-Inpainting` from https://github.com/researchmm/AOT-GAN-for-Inpainting at commit `2cd1afd8fdfabb101c678f6062d14bc7d302509e` (see `.gitmodules`). The `G0000000.pt` weights are not in Git.
- **Dataset files that came with the commit:** `data/waterbird_complete95_forest2water2/{metadata.csv,masks.csv,readme.txt}` and the Spawrious split files under `data/spawrious224/`. These are small metadata files, not images.
- **Licence:** none found in the upstream repository at this commit.
- **Modifications:** none to the copied files. Our memory patch is kept separately in `patches/`.
