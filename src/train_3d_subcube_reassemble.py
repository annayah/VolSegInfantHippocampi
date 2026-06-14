# Dependencies imports
import os
import sys
import argparse
import logging
import random
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from monai.losses import DiceCELoss

from model_3d import SegDINO3D, create_segdino3d
from dataset_3d import create_dataloaders, DATASET_STATS
from metrics_3d import (
    compute_dice,
    compute_iou,
    compute_volumetric_error,
)
from subcube_utils import split_volume, reassemble_volume


def setup_logger(name: str, log_dir: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict,
    path: str,
):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    torch.save(checkpoint, path)


# Reassemble then 2-pass loss training
def train_one_epoch_reassemble(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    sub_size: int = 64,
    scaler: Optional[GradScaler] = None,
    logger: Optional[logging.Logger] = None,
    use_amp: bool = True,
) -> Dict[str, float]:

    model.train()

    total_loss = 0.0
    dice_scores = []
    iou_scores = []

    pbar = tqdm(train_loader, desc=f"[Train Epoch {epoch}]")
    start_time = time.time()

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        _, _, d, h, w = images.shape

        sub_images, coords = split_volume(images, sub_size=sub_size)
        n_subs = len(sub_images)

        optimizer.zero_grad()

        # Pass 1: Forward all detached sub-cubes, reassemble them, compute the global loss
        detached_sub_preds = []

        for i in range(n_subs):
            with torch.no_grad():
                sub_pred = model(sub_images[i])
            # Store detached prediction (no graph, small memory footprint)
            detached_sub_preds.append(sub_pred.detach())

        # Reassemble into full volume - this is a plain tensor, no graph
        # But we need gradients w.r.t. this tensor, so we enable requires_grad
        full_pred = reassemble_volume(
            detached_sub_preds, coords,
            full_size=(d, h, w), sub_size=sub_size,
        )
        full_pred.requires_grad_(True)

        # Compute global loss on the full reassembled volume
        if use_amp and scaler is not None:
            with autocast():
                global_loss = criterion(full_pred, labels)
            # Get gradient of loss w.r.t. the full_pred tensor
            scaler.scale(global_loss).backward()
            # full_pred.grad now contains scaled gradients
            full_pred_grad = full_pred.grad.detach().clone()
            # We need to unscale these gradients for use in Pass 2
            # The scaler's current scale factor
            inv_scale = 1.0 / scaler.get_scale()
            full_pred_grad = full_pred_grad * inv_scale
        else:
            global_loss = criterion(full_pred, labels)
            global_loss.backward()
            full_pred_grad = full_pred.grad.detach().clone()

        batch_loss = global_loss.item()

        # Zero out the model gradients that may have been set, just to be safe
        optimizer.zero_grad()

        # Pass 2: Re-forward each sub-cube with the graph, backprop with upstream gradient
        for i in range(n_subs):
            d_s, h_s, w_s = coords[i]

            # Extract this sub-cube's portion of the global gradient
            sub_grad = full_pred_grad[
                :, :,
                d_s : d_s + sub_size,
                h_s : h_s + sub_size,
                w_s : w_s + sub_size,
            ].contiguous()

            # Re-forward this sub-cube (builds computational graph)
            if use_amp and scaler is not None:
                with autocast():
                    sub_pred = model(sub_images[i])
                # Scale the upstream gradient for the scaler
                scaled_sub_grad = sub_grad * scaler.get_scale()
                sub_pred.backward(gradient=scaled_sub_grad)
            else:
                sub_pred = model(sub_images[i])
                sub_pred.backward(gradient=sub_grad)

            # Graph for this sub-cube is freed after backward()

        # Optimizing step: Gradients from all 8 sub-cubes are accumulated
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += batch_loss

        # Monitoring metrics (reuse the detached full_pred from Pass 1)
        with torch.no_grad():
            # Recompute full_pred without grad tracking for metrics
            full_pred_metrics = reassemble_volume(
                detached_sub_preds, coords,
                full_size=(d, h, w), sub_size=sub_size,
            )
            pred_binary = (torch.sigmoid(full_pred_metrics) > 0.5).float()
            dice = compute_dice(pred_binary, labels).mean().item()
            iou = compute_iou(pred_binary, labels).mean().item()
            dice_scores.append(dice)
            iou_scores.append(iou)

        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'dice': f'{dice:.4f}',
            'iou': f'{iou:.4f}',
        })

    avg_loss = total_loss / len(train_loader)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)

    metrics = {'loss': avg_loss, 'dice': avg_dice, 'iou': avg_iou}

    epoch_time = time.time() - start_time
    metrics['time'] = epoch_time

    if logger:
        logger.info(
            f"[Train Epoch {epoch}] Loss: {avg_loss:.4f}, "
            f"Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f} "
            f"({epoch_time:.1f}s)"
        )

    return metrics


# Validation: Reassemble under no_gradient
@torch.no_grad()
def validate_subcube(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    sub_size: int = 64,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:

    model.eval()

    total_loss = 0.0
    dice_scores = []
    iou_scores = []
    vol_errors = []

    pbar = tqdm(val_loader, desc=f"[Val Epoch {epoch}]")
    start_time = time.time()

    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        spacing = batch['spacing'].to(device)

        _, _, d, h, w = images.shape

        sub_images, coords = split_volume(images, sub_size=sub_size)
        n_subs = len(sub_images)

        sub_preds = []
        for i in range(n_subs):
            sub_pred = model(sub_images[i])
            sub_preds.append(sub_pred)

        full_pred = reassemble_volume(
            sub_preds, coords,
            full_size=(d, h, w), sub_size=sub_size,
        )

        loss = criterion(full_pred, labels)
        total_loss += loss.item()

        pred_binary = (torch.sigmoid(full_pred) > 0.5).float()
        dice = compute_dice(pred_binary, labels).mean().item()
        iou = compute_iou(pred_binary, labels).mean().item()
        vol_error = compute_volumetric_error(pred_binary, labels, spacing).mean().item()

        dice_scores.append(dice)
        iou_scores.append(iou)
        vol_errors.append(vol_error)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
            'vol_err': f'{vol_error:.2f}%',
        })

    avg_loss = total_loss / len(val_loader)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_vol_error = np.mean(vol_errors)

    metrics = {
        'loss': avg_loss,
        'dice': avg_dice,
        'iou': avg_iou,
        'volumetric_error': avg_vol_error,
    }

    epoch_time = time.time() - start_time
    metrics['time'] = epoch_time

    if logger:
        logger.info(
            f"[Val Epoch {epoch}] Loss: {avg_loss:.4f}, "
            f"Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, "
            f"Vol Error: {avg_vol_error:.2f}% "
            f"({epoch_time:.1f}s)"
        )

    return metrics


# Training a fold
def train_fold(
    fold: int,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, float]:

    logger.info(f"{'='*60}")
    logger.info(f"[Reassemble] Training Fold {fold + 1}/{args.num_folds}")
    logger.info(f"{'='*60}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        rand_crop_size=tuple(args.rand_crop_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fold=fold,
        num_folds=args.num_folds,
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    model = create_segdino3d(
        dino_repo_path=args.dino_repo,
        dino_weights_path=args.dino_weights,
        encoder_size=args.encoder_size,
        nclass=args.num_classes,
        features=args.decoder_features,
        dino_input_size=args.dino_input_size,
    )
    model = model.to(device)
    model.encoder._freeze_backbone()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs,
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_epochs],
    )

    criterion = DiceCELoss(
        include_background=False, sigmoid=True,
        lambda_dice=0.5, lambda_ce=0.5,
    )

    scaler = GradScaler() if args.use_amp else None

    best_dice = 0.0
    best_metrics = {}
    patience_counter = 0
    fold_start_time = time.time()

    for epoch in range(1, args.epochs + 1):

        train_metrics = train_one_epoch_reassemble(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            sub_size=args.sub_size,
            scaler=scaler,
            logger=logger,
            use_amp=args.use_amp,
        )

        val_metrics = validate_subcube(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            sub_size=args.sub_size,
            logger=logger,
        )

        scheduler.step()

        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_metrics = val_metrics.copy()
            patience_counter = 0
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, metrics=val_metrics,
                path=os.path.join(args.output_dir, f'best_fold{fold}.pth'),
            )
            logger.info(f"New best model saved! Dice: {best_dice:.4f}")
        else:
            patience_counter += 1

        save_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, metrics=val_metrics,
            path=os.path.join(args.output_dir, f'latest_fold{fold}.pth'),
        )

        if args.patience > 0 and patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    fold_time = time.time() - fold_start_time
    best_metrics['fold_time'] = fold_time
    logger.info(f"Fold {fold + 1} Best Dice: {best_dice:.4f} (Total time: {fold_time/60:.1f}m)")
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description='Train 3D SegDINO - Reassemble Sub-Cubes')

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='albert', choices=['albert', 'lisa'])
    parser.add_argument('--rand_crop_size', type=int, nargs=3, default=[128, 128, 128])

    parser.add_argument('--dino_repo', type=str, required=True)
    parser.add_argument('--dino_weights', type=str, required=True)
    parser.add_argument('--encoder_size', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--dino_input_size', type=int, default=224)
    parser.add_argument('--decoder_features', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=1)

    parser.add_argument('--sub_size', type=int, default=64)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--use_amp', action='store_true')

    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=-1)

    parser.add_argument('--output_dir', type=str, default='./runs')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(
        args.output_dir,
        f'segdino3d_reassemble_{args.dataset}_{args.encoder_size}_{timestamp}'
    )
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logger('train', args.output_dir)
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")
    
    run_start_time = time.time()

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    all_metrics = []

    if args.fold >= 0:
        metrics = train_fold(args.fold, args, logger)
        all_metrics.append(metrics)
    else:
        for fold in range(args.num_folds):
            metrics = train_fold(fold, args, logger)
            all_metrics.append(metrics)

    total_run_time = time.time() - run_start_time
    logger.info("=" * 60)
    logger.info("Final Results (Cross-Validation)")
    logger.info(f"Total Run Time: {total_run_time/60:.1f}m")
    logger.info("=" * 60)

    dice_scores = [m['dice'] for m in all_metrics]
    iou_scores = [m['iou'] for m in all_metrics]
    vol_errors = [m['volumetric_error'] for m in all_metrics]

    logger.info(f"Dice: {np.mean(dice_scores):.4f} +/- {np.std(dice_scores):.4f}")
    logger.info(f"IoU: {np.mean(iou_scores):.4f} +/- {np.std(iou_scores):.4f}")
    logger.info(f"Vol Error: {np.mean(vol_errors):.2f}% +/- {np.std(vol_errors):.2f}%")

    results = {
        'approach': 'reassemble_then_loss',
        'sub_size': args.sub_size,
        'fold_metrics': all_metrics,
        'mean_dice': float(np.mean(dice_scores)),
        'std_dice': float(np.std(dice_scores)),
        'mean_iou': float(np.mean(iou_scores)),
        'std_iou': float(np.std(iou_scores)),
        'mean_vol_error': float(np.mean(vol_errors)),
        'std_vol_error': float(np.std(vol_errors)),
        'total_run_time_seconds': total_run_time,
    }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()