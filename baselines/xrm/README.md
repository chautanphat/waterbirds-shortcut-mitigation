# XRM upstream implementation

This directory preserves the official shared code from [facebookresearch/XRM](https://github.com/facebookresearch/XRM), pinned to commit `5e90815250aedea546b9e17ffd5d7e0a7c34c20f` for the ICML 2024 paper *Discovering Environments with XRM*.

The upstream code includes ERM, GroupDRO, and other algorithms. This project studies only the Waterbirds ERM and GroupDRO baselines. The original README is retained in [`UPSTREAM_README.md`](UPSTREAM_README.md); source, license, and local compatibility modifications are recorded in [`ORIGINAL_SOURCE.md`](ORIGINAL_SOURCE.md).

Use the separate method guides and project scripts:

- ERM: [`src/methods/erm/README.md`](../../src/methods/erm/README.md)
- GroupDRO: [`src/methods/groupdro/README.md`](../../src/methods/groupdro/README.md)

Project configs are in `configs/erm/` and `configs/groupdro/`. Keep project-specific behavior outside this upstream directory where possible.
