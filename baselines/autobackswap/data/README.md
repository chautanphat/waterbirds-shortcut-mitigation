# Datasets

## Waterbirds

Download links:
- **Main dataset:**  
  https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz

- **Segmentations:**  
  https://data.caltech.edu/records/w9d68-gec53


Folder structure:
```
data
└── waterbird_complete95_forest2water2
    ├── images
    │   ├── 001.Black_footed_Albatross
    │   └── ...
    ├──segmentations
    │   ├── 001.Black_footed_Albatross
    │   └── ...
    ├── masks.csv
    ├── metadata.csv
    └── readme.txt
```

---

## Spawrious

Download links:

- **Main dataset:**  
  https://github.com/aengusl/spawrious

- **SAM3 segmentations:** 

  Pre-generated segmentation files are not currently available for download. Instead, you can generate the segmentation masks yourself by running:
  [generate_segmentation_masks.py](generate_segmentation_masks.py)

Folder structure:
```
data
├── spawrious224
│   ├── m2m
│   │   ├── 0
│   │   ├── 1
│   │   ├── easy.json
│   │   ├── hard.json
│   │   └── medium.json
│   ├── o2o
│   │   ├── easy.json
│   │   ├── hard.json
│   │   └── medium.json
│   ├── masks.csv
│   └── readme.txt
└── spawrious224_segmentation_masks
    └── m2m
        ├── 0
        └── 1
```

> **Note:** 
In spawrious, the picture xxx/1/beach/labrador/beach_labrador_450.png is broken when downloading the original dataset, where xxx is o2o_easy o2o_medium, o2o_hard or m2m.
We cropped out a 224x224 image centered on the dogs face from the original broken image and replaced it.
Compare the [original](beach_labrador_450_orig.png) and the [replaced](beach_labrador_450.png) version of the image.
When running our experiments, replace the broken images with the cropped image accordingly.

---

## Spurious Vehicles

Download links:

- **Complete dataset**
  https://drive.google.com/file/d/1Ewwj6gkR0cCI50QvA1_zfF0PtjOcefc9/view?usp=sharing
- [Source code](generate_spur_vehicles.py)

Folder structure:
```
data
├── spurious_vehicles
│   ├── highway
│   ├── off-road
│   ├── parked
│   ├── rural
│   └── urban
└── spurious_vehicles_segmentation_masks
    ├── highway
    ├── off-road
    ├── parked
    ├── rural
    └── urban
```