# Dependencies imports
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
import scipy.ndimage

from dataset_3d import create_dataloaders, create_test_dataloader
from model_3d import create_segdino3d
from metrics_3d import compute_dice, compute_iou, compute_volumetric_error, compute_volume_ml
from subcube_utils import split_volume, reassemble_volume


# Visual - Selecting slices with max FG voxels
def pick_best_slices(gt: np.ndarray):
    """
    gt: [D, H, W]
    Returns (z, y, x) slices that maximize foreground per plane
    """
    x = np.argmax(gt.sum(axis=(1, 2)))  # saggital
    y = np.argmax(gt.sum(axis=(0, 2)))  # coronal
    z = np.argmax(gt.sum(axis=(0, 1)))  # axial
    return int(x), int(y), int(z)


# Visual
def _show_base(ax, img2d, title=None):
    ax.imshow(img2d, cmap="gray")
    if title:
        ax.set_title(title)
    ax.axis("off")


# Visual
def _overlay_mask(ax, mask2d, color, alpha=0.25, draw_contour=True):
    m = (mask2d > 0).astype(float)

    rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=float)
    
    r, g, b = mcolors.to_rgb(color)
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = m * alpha

    ax.imshow(rgba)

    if draw_contour:
        ax.contour(m, levels=[0.5], colors=[color], linewidths=1.5)


# Visual
def rot_left(img2d: np.ndarray) -> np.ndarray:
    return np.rot90(img2d, k=1)


# Qualitivative panel: 3x3 grid
def save_qual_panel(out_path, img, gt, pred, title):
    x, y, z = pick_best_slices(gt)

    # Axial: fix Z  -> (X, Y)
    axial_img = img[:, :, z]
    axial_gt  = gt[:, :, z]
    axial_pr  = pred[:, :, z]

    # Coronal: fix Y -> (X, Z)
    cor_img = img[:, y, :]
    cor_gt  = gt[:, y, :]
    cor_pr  = pred[:, y, :]

    # Sagittal: fix X -> (Y, Z)
    sag_img = img[x, :, :]
    sag_gt  = gt[x, :, :]
    sag_pr  = pred[x, :, :]

    # Axial
    axial_img = rot_left(axial_img)
    axial_gt  = rot_left(axial_gt)
    axial_pr  = rot_left(axial_pr)

    # Coronal
    cor_img = rot_left(cor_img)
    cor_gt  = rot_left(cor_gt)
    cor_pr  = rot_left(cor_pr)

    # Sagittal
    sag_img = rot_left(sag_img)
    sag_gt  = rot_left(sag_gt)
    sag_pr  = rot_left(sag_pr)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(title, fontsize=11)

    # Row 1: Axial
    _show_base(axes[0,0], axial_img, "Axial View")
    _show_base(axes[0,1], axial_img, "Ground Truth")
    _overlay_mask(axes[0,1], axial_gt, color="#00FF00", alpha=0.25)
    _show_base(axes[0,2], axial_img, "Prediction")
    _overlay_mask(axes[0,2], axial_pr, color="#FF0000", alpha=0.25)

    # Row 2: Coronal
    _show_base(axes[1,0], cor_img, "Coronal View")
    _show_base(axes[1,1], cor_img, "Ground Truth")
    _overlay_mask(axes[1,1], cor_gt, color="#00FF00", alpha=0.25)
    _show_base(axes[1,2], cor_img, "Prediction")
    _overlay_mask(axes[1,2], cor_pr, color="#FF0000", alpha=0.25)

    # Row 3: Sagittal
    _show_base(axes[2,0], sag_img, "Sagittal View")
    _show_base(axes[2,1], sag_img, "Ground Truth")
    _overlay_mask(axes[2,1], sag_gt, color="#00FF00", alpha=0.25)
    _show_base(axes[2,2], sag_img, "Prediction")
    _overlay_mask(axes[2,2], sag_pr, color="#FF0000", alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def infer_full_volume(model, image, sub_size):
    subs, coords = split_volume(image, sub_size=sub_size)
    preds = [model(s) for s in subs]
    d, h, w = image.shape[-3:]
    full_logits = reassemble_volume(preds, coords, full_size=(d, h, w), sub_size=sub_size)
    prob = torch.sigmoid(full_logits)
    pred_bin = (prob > 0.5).float()
    return full_logits, prob, pred_bin

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--checkpoint", default="best", choices=["best","latest"])
    ap.add_argument("--split", default="val", choices=["val","test"])
    ap.add_argument("--n_cases", type=int, default=8)
    ap.add_argument("--save_preproc_nifti", action="store_true", help="Save preprocessed image and GT alongside prediction")
    ap.add_argument("--save_original_space_nifti", action="store_true", help="Save prediction resampled back to original raw image space")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    args_json = run_dir / "args.json"
    if not args_json.exists():
        raise FileNotFoundError(f"Missing {args_json}")

    cfg = json.loads(args_json.read_text())
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")

    # Load model
    model = create_segdino3d(
        dino_repo_path=cfg["dino_repo"],
        dino_weights_path=cfg["dino_weights"],
        encoder_size=cfg["encoder_size"],
        nclass=cfg["num_classes"],
        features=cfg["decoder_features"],
        dino_input_size=cfg["dino_input_size"],
    ).to(device)

    ckpt_path = run_dir / f"{args.checkpoint}_fold{args.fold}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    # Loader
    if args.split == "test":
        loader = create_test_dataloader(cfg["data_dir"], cfg["dataset"], batch_size=1, num_workers=cfg["num_workers"])
    else:
        _, loader = create_dataloaders(
            data_dir=cfg["data_dir"],
            dataset_name=cfg["dataset"],
            rand_crop_size=tuple(cfg["rand_crop_size"]),
            batch_size=1,
            num_workers=cfg["num_workers"],
            fold=args.fold,
            num_folds=cfg["num_folds"],
        )

    out_dir = run_dir / "posthoc_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qual_panels").mkdir(exist_ok=True)
    (out_dir / "volume").mkdir(exist_ok=True)
    (out_dir / "predictions_nifti").mkdir(exist_ok=True)
    if args.save_original_space_nifti:
        (out_dir / "predictions_nifti_origspace").mkdir(exist_ok=True)

    rows = []
    count = 0
    eval_start_time = time.time()

    for batch in loader:
        if count >= args.n_cases:
            break

        case_id = batch.get("case_id", [f"case_{count:02d}"])[0]
        image_path = batch.get("image_path", [None])[0]
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        spacing = batch["spacing"].to(device)

        case_start_time = time.time()
        _, prob, pred_bin = infer_full_volume(model, image, sub_size=cfg["sub_size"])
        case_time = time.time() - case_start_time

        dice = compute_dice(pred_bin, label).mean().item()
        iou  = compute_iou(pred_bin, label).mean().item()
        ve   = compute_volumetric_error(pred_bin, label, spacing).mean().item()

        gt_ml   = compute_volume_ml(label, spacing).mean().item()
        pred_ml = compute_volume_ml(pred_bin, spacing).mean().item()
        abs_err_ml = abs(pred_ml - gt_ml)

        img_np = image[0,0].detach().cpu().numpy()
        gt_np  = (label[0,0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        pr_np  = (pred_bin[0,0].detach().cpu().numpy() > 0.5).astype(np.uint8)

        title = f"[{case_id}] Dice={dice:.3f}  IoU={iou:.3f}  VolErr={ve:.1f}%  GT={gt_ml:.2f}cm3  Pred={pred_ml:.2f}cm3  |Δ|={abs_err_ml:.2f}cm3"

        save_qual_panel(out_dir / "qual_panels" / f"{case_id}.png", img_np, gt_np, pr_np, title)

        # Save NIfTI prediction (preprocessed space)
        # Prefer the real post-transform affine (encodes Spacing + Crop + Pad + CenterCrop).
        # Fallback to a synthetic spacing-only affine if it's not in the batch.
        if "preproc_affine" in batch:
            affine = batch["preproc_affine"][0].cpu().numpy().astype(np.float64)
        else:
            sx, sy, sz = spacing[0].cpu().numpy()
            affine = np.diag([-sx, -sy, sz, 1.0])
            
        pred_nifti = nib.Nifti1Image(pr_np, affine)
        nib.save(pred_nifti, out_dir / "predictions_nifti" / f"{case_id}_pred.nii.gz")
        
        if args.save_preproc_nifti:
            img_nifti = nib.Nifti1Image(img_np, affine)
            gt_nifti = nib.Nifti1Image(gt_np, affine)
            nib.save(img_nifti, out_dir / "predictions_nifti" / f"{case_id}_image.nii.gz")
            nib.save(gt_nifti, out_dir / "predictions_nifti" / f"{case_id}_gt.nii.gz")

        # Save NIfTI prediction (original raw image space)
        if args.save_original_space_nifti and image_path is not None and os.path.exists(image_path):
            orig_img = nib.load(image_path)
            orig_affine = orig_img.affine
            orig_shape = orig_img.shape[:3]
            
            # To map the 128^3 prediction back to the original grid without tracking all MONAI transforms, we can use scipy.ndimage.affine_transform.
            # We need the transformation matrix from the original voxel space to the preprocessed voxel space.
            # T_orig2world = orig_affine
            # T_preproc2world = affine
            # T_orig2preproc = inv(T_preproc2world) @ T_orig2world
            
            try:
                inv_affine = np.linalg.inv(affine)
                T_orig2preproc = inv_affine @ orig_affine
                
                # scipy.ndimage.affine_transform expects the matrix mapping output (orig) to input (preproc)
                # which is exactly T_orig2preproc.
                # We extract the 3x3 rotation/scale matrix and the translation vector.
                matrix = T_orig2preproc[:3, :3]
                offset = T_orig2preproc[:3, 3]
                
                # Resample the binary prediction
                pr_np_float = pr_np.astype(np.float32)
                pr_orig_space = scipy.ndimage.affine_transform(
                    pr_np_float,
                    matrix=matrix,
                    offset=offset,
                    output_shape=orig_shape,
                    order=1, # Trilinear interpolation
                    mode='constant',
                    cval=0.0
                )
                
                # Threshold back to binary
                pr_orig_space_bin = (pr_orig_space > 0.5).astype(np.uint8)
                
                orig_pred_nifti = nib.Nifti1Image(pr_orig_space_bin, orig_affine)
                nib.save(orig_pred_nifti, out_dir / "predictions_nifti_origspace" / f"{case_id}_pred_origspace.nii.gz")
            except Exception as e:
                print(f"Warning: Failed to resample {case_id} to original space: {e}")

        rows.append([case_id, dice, iou, ve, gt_ml, pred_ml, abs_err_ml, case_time])
        count += 1

    total_eval_time = time.time() - eval_start_time

    # Save per-case CSV
    csv_path = out_dir / "per_case_metrics.csv"
    with open(csv_path, "w") as f:
        f.write("case_id,dice,iou,vol_error_percent,gt_ml,pred_ml,abs_err_ml,inference_time_s\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f},{r[7]:.4f}\n")

    # Volume plots
    if len(rows) > 0:
        gt = np.array([r[4] for r in rows], dtype=float)
        pr = np.array([r[5] for r in rows], dtype=float)

        # Scatter pred vs gt with y=x
        plt.figure()
        plt.scatter(gt, pr)
        mn = min(gt.min(), pr.min())
        mx = max(gt.max(), pr.max())
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel("GT volume (cm3)")
        plt.ylabel("Pred volume (cm3)")
        plt.title("Pred vs GT volume")
        plt.tight_layout()
        plt.savefig(out_dir / "volume" / "volume_scatter.png", dpi=200)
        plt.close()

        # Bland–Altman
        mean = (gt + pr) / 2.0
        diff = (pr - gt)
        md = diff.mean()
        sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
        loa1 = md - 1.96 * sd
        loa2 = md + 1.96 * sd

        plt.figure()
        plt.scatter(mean, diff)
        plt.axhline(md, linestyle="--")
        plt.axhline(loa1, linestyle="--")
        plt.axhline(loa2, linestyle="--")
        plt.xlabel("Mean volume (cm3)")
        plt.ylabel("Pred - GT (cm3)")
        plt.title("Bland–Altman (cm3)")
        plt.tight_layout()
        plt.savefig(out_dir / "volume" / "bland_altman.png", dpi=200)
        plt.close()

        # Correlations + MAE
        mae = np.mean(np.abs(diff))
        # Pearson
        pear = np.corrcoef(gt, pr)[0,1] if len(gt) > 1 else np.nan
        # Spearman
        def rankdata(x):
            temp = x.argsort()
            ranks = np.empty_like(temp, dtype=float)
            ranks[temp] = np.arange(len(x), dtype=float)
            return ranks
        spear = np.corrcoef(rankdata(gt), rankdata(pr))[0,1] if len(gt) > 1 else np.nan

        stats_txt = out_dir / "volume" / "volume_stats.txt"
        stats_txt.write_text(
            f"n={len(gt)}\nMAE(cm3)={mae:.4f}\nPearson r={pear:.4f}\nSpearman rho={spear:.4f}\n"
            f"Total Eval Time: {total_eval_time:.1f}s\n"
        )

    print(f"Saved figures + CSV to: {out_dir}")
    print(f"Total evaluation time: {total_eval_time:.1f}s")

if __name__ == "__main__":
    main()