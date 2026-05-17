# AI-driven Facial Landmark Localisation on 3D Facial Scans

This repository contains the public code release for a master's thesis project on automatic 3D facial landmark localisation.

The goal is to predict nine anatomical facial landmarks from 3D facial point clouds using deep learning.

## Task

Input:

- 3D facial point cloud
- Optional facial mesh for surface projection

Output:

- 9 anatomical landmarks
- 3D coordinates for each landmark
- 27 output values in total

The landmarks are:

1. Glabella
2. Nasion
3. Rhinion
4. Nasal Tip
5. Subnasale
6. Alare Right
7. Alare Left
8. Zygion Right
9. Zygion Left

## Method

The main method is a coarse-to-fine framework called **Point Transformer C2F**.

The pipeline is:

1. A Point Transformer predicts coarse landmark locations from the full facial point cloud.
2. Local patches are cropped around the coarse predictions.
3. A PointNet++ heatmap refiner predicts local landmark likelihoods.
4. Soft-argmax decoding produces refined landmark coordinates.
5. Predictions can be evaluated before snapping, snapped to the point cloud, or projected to the facial mesh.

## Repository Structure

```text
models/          Model definitions
scripts/         Training, evaluation, and preprocessing scripts
utils/           Utility functions

README.md        Project description
```

## Data

The private 3D facial scans, landmark annotations, patient information, trained weights, and internal result files are not included.

To use this code, prepare your own data in a similar format:

```text
data/
├── sample_001/
│   ├── pointcloud.npy
│   ├── landmarks.npy
│   └── mesh file, optional
├── sample_002/
│   ├── pointcloud.npy
│   ├── landmarks.npy
│   └── mesh file, optional
```

The point cloud should have shape `N x 3`.

The landmark annotation should have shape `9 x 3`.

## Installation

```bash
conda create -n facial_landmarks python=3.11
conda activate facial_landmarks
pip install -r requirements.txt
```

## Training

Example command:

```bash
python scripts/training/train_point_transformer_c2f.py \
   --config configs/example_config.yaml \
   --data_dir data/processed \
   --output_dir results/point_transformer_c2f
```

The exact command may need to be adjusted depending on the released script names.

## Evaluation

Example command:

```bash
python scripts/evaluation/evaluate_model.py \
   --config configs/example_config.yaml \
   --checkpoint results/point_transformer_c2f/best_model.pth \
   --data_dir data/processed \
   --output_dir results/evaluation
```

Evaluation is reported in millimetres using 3D Euclidean landmark error.

Supported output variants:

* `before-snapping`: raw model prediction
* `snap-pc`: prediction snapped to the sampled point cloud
* `snap-mesh`: prediction projected to the facial mesh surface

## Reproducibility

The thesis experiments used:

* patient-level data splitting
* training-side model selection
* fixed random seed
* point-cloud normalisation without test-label leakage
* final holdout evaluation

When reproducing the experiments, report the dataset split, number of sampled points, model configuration, random seed, and evaluation metric.

## Limitations

This code is provided for research purposes only.

It is not a clinically validated tool and should not be used for diagnosis, treatment planning, or clinical decision-making without further validation.

## Citation

If you use this code, please cite the associated thesis:

```text
Hu, J. and Hou, Y. AI-driven Facial Landmark Localisation on 3D Facial Scans.
Master's Thesis, KU Leuven, Faculty of Engineering Technology, 2025–2026.
```

## License

The license should be specified before public release.

The dataset is not included and is subject to its own access restrictions.
