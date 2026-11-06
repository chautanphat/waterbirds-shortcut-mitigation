Spawrious folder structure:

 ── spawrious224
    ├── m2m
    │   ├── 0  <-- image files for env-0
    │   ├── 1  <-- image files for env-1
    │   ├── easy.json
    │   ├── hard.json
    │   └── medium.json
    ├── o2o  <-- no images stored here
    │   ├── easy.json
    │   ├── hard.json
    │   └── medium.json
    └── masks.csv  <-- patch-level mask annotations

NOTE:
Spawrious provides separate folders for O2O and M2M splits,
but the underlying images used for O2O are a subset of the M2M data.
Therefore downloading both leads to duplicated images.
To avoid having duplicate data, this implementation uses only the M2M folders
as the physical data source. All splits are defined logically via metadata.
