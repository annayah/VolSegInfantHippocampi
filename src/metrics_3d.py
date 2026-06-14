# Dependencies imports
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Custom metric functions
def compute_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    threshold: float = 0.5,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    
    if apply_sigmoid:
        pred = torch.sigmoid(pred)
    
    pred = (pred > threshold).float()
    target = target.float()

    if pred.dim() == 5:
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target.view(target.shape[0], target.shape[1], -1)
    elif pred.dim() == 4:
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target.view(target.shape[0], target.shape[1], -1)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {pred.dim()}D")
    
    intersection = (pred_flat * target_flat).sum(dim=-1)
    union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1)

    dice = (2.0 * intersection + eps) / (union + eps)
    
    if dice.shape[1] > 1:
        dice = dice.mean(dim=1)
    else:
        dice = dice.squeeze(1)
    
    return dice


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    threshold: float = 0.5,
    apply_sigmoid: bool = False,
) -> torch.Tensor:

    if apply_sigmoid:
        pred = torch.sigmoid(pred)

    pred = (pred > threshold).float()
    target = target.float()
    
    if pred.dim() == 5:
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target.view(target.shape[0], target.shape[1], -1)
    elif pred.dim() == 4:
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target.view(target.shape[0], target.shape[1], -1)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {pred.dim()}D")
    
    intersection = (pred_flat * target_flat).sum(dim=-1)
    union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1) - intersection
    
    iou = (intersection + eps) / (union + eps)
    
    if iou.shape[1] > 1:
        iou = iou.mean(dim=1)
    else:
        iou = iou.squeeze(1)
    
    return iou


def compute_volumetric_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    spacing: torch.Tensor,
    threshold: float = 0.5,
    apply_sigmoid: bool = False,
) -> torch.Tensor:

    if apply_sigmoid:
        pred = torch.sigmoid(pred)
    
    pred = (pred > threshold).float()
    target = target.float()
    
    pred_voxels = pred.sum(dim=(1, 2, 3, 4))
    target_voxels = target.sum(dim=(1, 2, 3, 4))
    
    voxel_volume = spacing[:, 0] * spacing[:, 1] * spacing[:, 2]
    
    pred_volume = pred_voxels * voxel_volume
    target_volume = target_voxels * voxel_volume
    
    vol_error = torch.abs(pred_volume - target_volume) / (target_volume + 1e-6) * 100
    
    return vol_error


def compute_volume_ml(
    mask: torch.Tensor,
    spacing: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:

    mask = (mask > threshold).float()

    voxel_count = mask.sum(dim=(1, 2, 3, 4))

    voxel_volume = spacing[:, 0] * spacing[:, 1] * spacing[:, 2]

    volume_ml = voxel_count * voxel_volume / 1000.0
    
    return volume_ml


class MetricTracker:

    def __init__(self, metrics: List[str] = ['dice', 'iou', 'volumetric_error']):
        self.metrics = metrics
        self.reset()
    
    def reset(self):
        self._values = {m: [] for m in self.metrics}
        self._running_sum = {m: 0.0 for m in self.metrics}
        self._count = 0
    
    def update(self, values: Dict[str, float], n: int = 1):

        for metric in self.metrics:
            if metric in values:
                self._values[metric].append(values[metric])
                self._running_sum[metric] += values[metric] * n
        self._count += n
    
    def get_mean(self, metric: str) -> float:
        if self._count == 0:
            return 0.0
        return self._running_sum[metric] / self._count
    
    def get_all_means(self) -> Dict[str, float]:
        return {m: self.get_mean(m) for m in self.metrics}
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for metric in self.metrics:
            values = np.array(self._values[metric])
            if len(values) > 0:
                summary[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
            else:
                summary[metric] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                }
        return summary
    
    def get_values(self, metric: str) -> List[float]:
        return self._values[metric]
    

# Batch metric computation
def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    spacing: torch.Tensor,
    threshold: float = 0.5,
    apply_sigmoid: bool = False,
) -> Dict[str, torch.Tensor]:

    metrics = {
        'dice': compute_dice(pred, target, threshold=threshold, apply_sigmoid=apply_sigmoid),
        'iou': compute_iou(pred, target, threshold=threshold, apply_sigmoid=apply_sigmoid),
        'volumetric_error': compute_volumetric_error(pred, target, spacing, threshold=threshold, apply_sigmoid=apply_sigmoid),
    }
    
    if apply_sigmoid:
        pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    
    metrics['pred_volume_ml'] = compute_volume_ml(pred_binary, spacing)
    metrics['target_volume_ml'] = compute_volume_ml(target, spacing)
    
    return metrics


# Metrics (again) numpy-based for post-processing
def compute_dice_np(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-6,
) -> float:

    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(dice)


def compute_iou_np(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-6,
) -> float:

    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    iou = (intersection + eps) / (union + eps)
    return float(iou)


def compute_volumetric_error_np(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float, float],
) -> float:

    pred = pred.astype(bool)
    target = target.astype(bool)
    
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    
    pred_volume = pred.sum() * voxel_volume
    target_volume = target.sum() * voxel_volume
    
    if target_volume < 1e-6:
        return 0.0 if pred_volume < 1e-6 else 100.0
    
    vol_error = abs(pred_volume - target_volume) / target_volume * 100
    return float(vol_error)