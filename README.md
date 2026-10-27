# DuetDP-DD

This repository is a modified version of the official DPImageBench codebase. It keeps the original DPImageBench framework for differentially private image synthesis, and adds several scripts and evaluation components for training classifiers and evaluating distilled data utility.

The original DPImageBench project should be used as the reference for environment installation, dataset preparation, and the general benchmark workflow.

## Based On

This project is developed based on:

- DPImageBench: A Unified Benchmark for Differentially Private Image Synthesis Algorithms
- Original project/paper: please refer to the official DPImageBench repository and paper for the complete benchmark description, supported methods, datasets, and citation information.

This repository only documents the modifications and additional usage introduced in this version.

## Main Changes

Compared with the official DPImageBench codebase, this version includes the following changes:

1. Added `train_dp_classifier.py`

   This script is used to train a `dp_classifier` on synthetic data.

2. Added `train_semantic_anchor_classifier.py`

   This script is used to train a `semantic_anchor_classifier`. It enables hierarchical/semantic public data loading before classifier training.

3. Added `eval_distilled_data_utility.py`

   This script is used to evaluate the utility of distilled data. It loads generated/distilled samples and evaluates their downstream classification utility through the evaluation pipeline.

4. Modified the `evaluation/` module

   The `evaluation/evaluator.py` file has been adjusted to support the added classifier training and distilled data utility evaluation workflow.

   The classifier folder has also been extended with additional models:

   - `evaluation/classifier/resnet_18.py`
   - `evaluation/classifier/dense121.py`


## Installation

Please follow the official DPImageBench instructions to install the environment and dependencies.

The general process is:

```bash
conda create -n dpimagebench python=3.9
conda activate dpimagebench
pip install -r requirements.txt
```

Depending on the method and dataset used, additional dependencies may be required. Please refer to the original DPImageBench documentation for details.

## Data Preparation

Please prepare datasets following the official DPImageBench data preparation instructions.

The added scripts assume that the DPImageBench data loading pipeline is already correctly configured. In particular, dataset paths, method configurations, and generated sample paths should be checked before running the scripts.

## Generate DP Synthetic Dataset

Before training classifiers or evaluating distilled data, first run the original DPImageBench synthesis pipeline to generate a differentially private synthetic dataset.

For example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
  setup.n_gpus_per_node=4 \
  --method PrivImage \
  --data_name cifar10_32 \
  --epsilon 1.0
```

If your available GPU IDs are different, change `CUDA_VISIBLE_DEVICES` accordingly.

This step produces synthetic samples under the experiment output directory, such as:

```text
./exp/privimage/cifar10_32_eps1.0_synthetic_dataset/gen/gen.npz
```

The following classifier training and utility evaluation scripts use this generated `.npz` file as input by default. If your generated dataset is saved to another path, update `gen_path` in the corresponding script before running it.

## Usage

After completing environment installation, dataset preparation, and DP synthetic dataset generation according to DPImageBench, run the added scripts from the project root.

### Train DP Classifier

```bash
python train_dp_classifier.py \
  --method PrivImage \
  --data_name cifar10_32 \
  --epsilon 1.0
```

By default, the script loads synthetic samples from:

```text
./exp/privimage/cifar10_32_eps1.0_synthetic_dataset/gen/gen.npz
```

Please update the path in the script if your generated data is stored elsewhere.

### Train Semantic Anchor Classifier

```bash
python train_semantic_anchor_classifier.py \
  --method PrivImage \
  --data_name cifar10_32 \
  --epsilon 1.0
```

This script enables semantic data loading and then trains the classifier using the semantic data the and generated synthetic samples.

### Evaluate Distilled Data Utility

```bash
python eval_distilled_data_utility.py \
  --method PrivImage \
  --data_name cifar10_32 \
  --epsilon 1.0
```

This script evaluates the utility of distilled data through the modified evaluator.

The default generated data path is:

```text
./exp/privimage/cifar10_32_eps1.0_synthetic_dataset/gen/gen.npz
```

The default classifier/result save path is:

```text
./exp/privimage/cifar10_32_eps1.0_synthetic_dataset
```

Modify these paths in the script if necessary.

## Repository Structure

Important files and folders in this modified version:

```text
.
+-- configs/                              # DPImageBench configuration files
+-- data/                                 # Dataset loading and preprocessing code
+-- evaluation/
|   +-- evaluator.py                      # Modified evaluator
|   +-- classifier/
|       +-- dense121.py                   # Added classifier model
|       +-- resnet_18.py                  # Added classifier model
|       +-- ...
+-- exp/
|   +-- privimage/                        # Example synthetic sample output
+-- models/                               # DPImageBench model implementations
+-- train_dp_classifier.py                # Added DP classifier training script
+-- train_semantic_anchor_classifier.py   # Added semantic anchor classifier training script
+-- eval_distilled_data_utility.py        # Added distilled data utility evaluation script
```

## Notes Before Public Release

Before uploading this repository to GitHub, please carefully check:

- Whether generated data under `exp/` should be uploaded.
- Whether model checkpoints, datasets, logs, or large binary files are included.
- Whether all third-party licenses are preserved.
- Whether the original DPImageBench project is properly credited.

Recommended files to exclude from Git tracking include:

```text
exp/
dataset/
logs/
lightning_logs/
*.ckpt
*.pth
*.pt
*.pkl
*.npz
*.npy
__pycache__/
*.pyc
.idea/
```

If a small synthetic sample is intentionally provided as an example, please state its purpose clearly and make sure it is allowed to be redistributed.

## License and Acknowledgement

This repository is based on DPImageBench and may include third-party components from other open-source projects. Please retain the original license files and copyright notices when redistributing the code.

If you use this repository, please also cite and acknowledge the original DPImageBench project and any third-party projects used by the corresponding modules.
