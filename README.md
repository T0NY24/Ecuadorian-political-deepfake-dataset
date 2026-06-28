---
license: mit
task_categories:
  - image-classification
language:
  - es
tags:
  - deepfake
  - detection
  - ecuador
  - political
  - computer-vision
  - FLUX
  - synthetic-media
size_categories:
  - 1K<n<10K
---

# Ecuadorian Political Deepfake Dataset

## Dataset Description

A balanced dataset of **1,462 facial images** of four Ecuadorian
political figures, designed for deepfake detection research.

- **Real images:** 731
- **Synthetic images:** 731 (generated with FLUX.1-dev + LoRA)

## Political Figures

| Politician       | Real | Fake | Total |
|-----------------|------|------|-------|
| Daniel Noboa    | 231  | 231  | 462   |
| Rafael Correa   | 100  | 100  | 200   |
| Guillermo Lasso | 200  | 200  | 400   |
| Luisa Gonzalez  | 200  | 200  | 400   |
| **Total**       | **731** | **731** | **1,462** |

## Generation Method

- **Base model:** FLUX.1-dev
- **Fine-tuning:** LoRA (rank=32, alpha=32)
- **Platform:** MimicPC + FluxGym
- **Inference:** ComfyUI (28 steps, CFG 7.0)
- **Resolution:** 512x512 px (preprocessing), 1024x1024 (generation)

## Preprocessing

- **Face detection:** MTCNN (confidence threshold: 0.85)
- **ROI expansion:** 1.8w x 1.9h
- **Normalization:** Bicubic scaling to 512x512 px

## Dataset Splits

| Split      | Images | Percentage |
|-----------|--------|------------|
| Train     | 1,023  | ~70%       |
| Validation| 219  | ~15%       |
| Test      | 220  | ~15%       |

## Directory Structure

```
ecuadorian-political-deepfake-dataset/
├── README.md
├── metadata.csv
├── dataset_info.json
└── data/
    ├── real/
    │   ├── noboa/        (231 images)
    │   ├── correa/       (100 images)
    │   ├── lasso/        (200 images)
    │   └── gonzalez/     (200 images)
    └── fake/
        ├── noboa/        (231 images)
        ├── correa/       (100 images)
        ├── lasso/        (200 images)
        └── gonzalez/     (200 images)
```

## Metadata CSV Columns

| Column     | Description                                       |
|-----------|---------------------------------------------------|
| file_path  | Relative path to the image                       |
| label      | `real` or `fake`                                  |
| politician | Short key: `noboa`, `correa`, `lasso`, `gonzalez` |
| split      | `train`, `val`, or `test`                         |
| source     | `web_scraping`, `video_extraction`, `FLUX1_LoRA`  |
| resolution | Image resolution (`512x512`)                      |

## Citation

If you use this dataset, please cite:

```
[Citation will be added after publication - TICEC 2026]
```

## License

MIT License - See LICENSE file for details.
