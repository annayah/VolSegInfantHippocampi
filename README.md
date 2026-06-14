## Volumetric Segmentation of the Infant Hippocampus in Brain MRIs by Adapting Foundation Models

This repository contains the research code for a project on **3D infant hippocampus segmentation from brain MRI** using a frozen DINOv3-style vision transformer encoder and a lightweight 3D decoder. The project investigates whether strong 2D visual foundation-model representations can be adapted to volumetric medical image segmentation through slice-wise encoding, depth-aware reboxing, and sub-cube based full-volume training.

The central methodological question is how to adapt a powerful 2D foundation model to a 3D infant neuroimaging task while preserving volumetric consistency and keeping the training pipeline feasible on limited GPU memory.

The codebase is organized for academic review, reproducibility, and defense presentation. It includes dataset utilities, model definition, sub-cube splitting and reassembly, training, metric computation, post-hoc evaluation, and plotting scripts.

## Project overview

The implementation follows a two-level design. At the **methodological level**, a 3D MRI volume is represented as a stack of 2D slices, encoded through a frozen transformer backbone, reshaped back into depth-aware 3D feature maps, and decoded into a hippocampus segmentation mask. At the **software level**, the repository separates data preparation, model construction, training orchestration, metrics, and visualization into distinct scripts.

| Aspect | Description |
|---|---|
| Task | Binary hippocampus segmentation from infant brain MRI volumes. |
| Core idea | Adapt a frozen 2D DINOv3-style encoder to 3D segmentation using slice-wise encoding and depth-aware volumetric decoding. |
| Datasets used | ALBERT and LISA infant MRI datasets; data are not redistributed in this repository. |
| Main training script | `src/train_3d_subcube_reassemble.py` |
| Model definition | `src/model_3d.py` |
| Evaluation outputs | Dice, IoU, relative volume error, physical volume, qualitative panels, NIfTI masks, and learning curves. |

## Repository structure

```text
segdino3d/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── src/
│   ├── dataset_3d.py
│   ├── model_3d.py
│   ├── subcube_utils.py
│   ├── train_3d_subcube_reassemble.py
│   ├── metrics_3d.py
│   ├── posthoc_eval_figures.py
│   └── plot_learning_curves.py
```
## Major code components

| File | Primary responsibility | Theoretical method link |
|---|---|---|
| `src/dataset_3d.py` | Defines MONAI-based preprocessing, dataset statistics, label binarization, channel repetition, cross-validation splitting, and dataloaders. | Converts heterogeneous NIfTI MRI volumes into standardized tensors for model training. |
| `src/model_3d.py` | Defines `SegDINO3DEncoder`, `DPTHead3D`, `SegDINO3D`, and the model factory. | Implements frozen 2D encoder adaptation, depth-aware reboxing, and 3D decoding. |
| `src/subcube_utils.py` | Splits full 3D volumes into sub-cubes and reassembles sub-cube predictions. | Enables memory-aware training and inference on volumetric images. |
| `src/train_3d_subcube_reassemble.py` | Orchestrates training, validation, checkpointing, logging, early stopping, AMP, and fold control. | Operationalizes the full workflow. |
| `src/metrics_3d.py` | Computes Dice, IoU, relative volume error, and physical volume. | Provides quantitative evaluation of segmentation quality and volume agreement. |
| `src/posthoc_eval_figures.py` | Runs full-volume inference, saves qualitative panels, NIfTI masks, CSV summaries, and additional analyses. | Supports post-hoc inspection and defense figures. |
| `src/plot_learning_curves.py` | Parses training logs and plots loss, Dice, and IoU curves. | Produces convergence and monitoring figures. |

## Dependencies

The code was prepared for Python 3.11 and PyTorch-based training. A GPU-enabled PyTorch installation is recommended for practical experiments. If your CUDA version requires a specific PyTorch wheel, install PyTorch first using the official selector and then install the remaining dependencies from `requirements.txt`.[1] An A40 GPU was utilized from RunPod.

## Data preparation

The project used infant MRI datasets that are not included in the repository. The training scripts expect a dataset directory containing image and label volumes in NIfTI-compatible formats. A recommended local-only organization is shown below.

```text
data/
├── albert/
│   ├── images/
│   └── labels/
└── lisa/
    ├── images/
    └── labels/
```

Because neuroimaging datasets may contain sensitive or access-controlled research data, do not commit raw MRI volumes, masks, or derived subject-level outputs to GitHub. Keep them locally, or use an institution-approved storage location.

## Training

The main training entry point is `src/train_3d_subcube_reassemble.py`. The command-line interface exposes dataset choice, DINO repository and checkpoint paths, encoder size, crop size, sub-cube size, optimizer settings, fold selection, mixed precision, and output location.

```bash
python src/train_3d_subcube_reassemble.py \
  --data_dir /path/to/data \
  --dataset albert \
  --dino_repo /path/to/dinov3 \
  --dino_weights /path/to/dinov3_weights.pth \
  --encoder_size base \
  --rand_crop_size 128 128 128 \
  --sub_size 64 \
  --epochs 100 \
  --batch_size 1 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --num_folds 5 \
  --fold 0 \
  --use_amp \
  --output_dir runs/albert_base_fold0
```

To train all folds, set `--fold -1`. To switch datasets, change `--dataset lisa` and point `--data_dir` to the appropriate dataset root.

## Post-hoc evaluation and figures

After training, use the post-hoc script to generate qualitative panels, saved prediction masks, and metric summaries. The exact arguments should match your trained checkpoint and dataset location.

```bash
python src/posthoc_eval_figures.py \
  --data_dir /path/to/data \
  --dataset albert \
  --checkpoint runs/albert_base_fold0/best_model.pth \
  --dino_repo /path/to/dinov3 \
  --dino_weights /path/to/dinov3_weights.pth \
  --output_dir results/albert_base_fold0_posthoc
```

Training curves can be generated from log files using:

```bash
python src/plot_learning_curves.py \
  --log_path runs/albert_base_fold0/train.log \
  --output_dir figures/albert_base_fold0
```

## Reproducibility notes

The training script includes seed control, fold selection, checkpointing, and structured logging. However, exact numerical reproducibility may still depend on GPU type, CUDA/cuDNN behavior, PyTorch version, data preprocessing details, and the exact DINO weights used. For a thesis defense or paper artifact, record the full environment, dataset split files, checkpoint hashes, and command-line arguments for each reported experiment.

## Citation

If this repository is used as part of academic evaluation, cite the report and this code repository. A draft `CITATION.cff` file is included and should be updated with the final repository URL, release version, and submission details before publication.

## References

[1]: https://pytorch.org/get-started/locally/ "PyTorch: Get Started"
[2]: https://monai.io/ "MONAI: Medical Open Network for AI"
[3]: https://github.com/facebookresearch/dinov3 "DINOv3 repository"
