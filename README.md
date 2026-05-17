# Public Code Release

This folder contains a cleaned code-only subset of the thesis project.
Private data, checkpoints, generated results, and local machine artifacts are intentionally excluded.

## Included components

### Core proposed method

- scripts/training/train_heatmap_joint_hybrid_pt_patch_floor_v1_kfold.py
- scripts/training/train_heatmap_joint_flip_v3.py
- scripts/training/train_heatmap_joint_flip_v5_cascade_soft_roi.py

### Baseline training scripts used in final comparisons

- scripts/training/main_script_pointtransformer_kfold.py
- scripts/training/main_script_pointnet2_kfold.py

### Evaluation scripts for raw / snap-point / snap-mesh metrics

- scripts/eval_v1_inner_ensemble.py
- scripts/eval_f02_snap_pt.py
- scripts/eval_pointnet2_snap.py

### Model definitions and local utility modules

- models/point_transformer_reg.py
- models/pointnet2_reg.py
- scripts/utils/pointnet2_utils.py
- scripts/utils/pointnet_utils.py

### Optional PointNet++ CUDA extension source

- pointnet2_ops_lib/setup.py
- pointnet2_ops_lib/pointnet2_ops/__init__.py
- pointnet2_ops_lib/pointnet2_ops/_version.py
- pointnet2_ops_lib/pointnet2_ops/pointnet2_modules.py
- pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py
- pointnet2_ops_lib/pointnet2_ops/_ext-src/include/ball_query.h
- pointnet2_ops_lib/pointnet2_ops/_ext-src/include/cuda_utils.h
- pointnet2_ops_lib/pointnet2_ops/_ext-src/include/group_points.h
- pointnet2_ops_lib/pointnet2_ops/_ext-src/include/interpolate.h
- pointnet2_ops_lib/pointnet2_ops/_ext-src/include/sampling.h
- pointnet2_ops_lib/pointnet2_ops/_ext-src/include/utils.h
- pointnet2_ops_lib/pointnet2_ops/_ext-src/src/ball_query.cpp
- pointnet2_ops_lib/pointnet2_ops/_ext-src/src/ball_query_gpu.cu
- pointnet2_ops_lib/pointnet2_ops/_ext-src/src/bindings.cpp
- pointnet2_ops_lib/pointnet2_ops/_ext-src/src/group_points.cpp
- pointnet2_ops_lib/pointnet2_ops/_ext-src/src/group_points_gpu.cu
- pointnet2_ops_lib/pointnet2_ops/_ext-src/src/interpolate.cpp
- pointnet2_ops_lib/pointnet2_ops/_ext-src/src/interpolate_gpu.cu
- pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling.cpp
- pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu

## Environment setup

1. Create a Python environment with Python 3.10+.
2. Install the main Python packages used by the release scripts:
   - torch
   - numpy
   - scikit-learn
   - tqdm
3. If you want the optimized PointNet++ operators, build the package in `pointnet2_ops_lib/`.
   The training code also contains a fallback path that can run without the compiled extension, although it may be slower.

## Expected data layout

Place processed samples under `data/pointcloud/<sample_name>/` with the following files:
- `pointcloud_full.npy`
- `nose_landmarks.npy`
- `triangles.npy` (optional but recommended for mesh snapping)

The release does not include any dataset files.

## Train the proposed Point Transformer coarse-to-fine model

Run the main training script from the project root:
- `python scripts/training/train_heatmap_joint_hybrid_pt_patch_floor_v1_kfold.py --seed 42 --sigma 0.04`

This script contains the final unified PT coarse localisation stage, the PointNet++ patch refiner, data loading, preprocessing, training, and holdout evaluation logic.

## Evaluate the final model

Examples:
- `python scripts/eval_v1_inner_ensemble.py` for inner-fold ensemble vs final retrain evaluation.
- `python scripts/eval_f02_snap_pt.py` for the Point Transformer baseline raw / snap-point / snap-mesh metrics.
- `python scripts/eval_pointnet2_snap.py` for the PointNet++ baseline raw / snap-point / snap-mesh metrics.

## Notes

- Checkpoints are not included in this release.
- Result tables can be regenerated from the JSON or console summaries produced by the training and evaluation scripts.
- See `MANIFEST.txt` for the exact file list copied into this folder.

Generated on 2026-05-17 19:21 UTC
