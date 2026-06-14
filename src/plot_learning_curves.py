# Dependencies imports
import argparse, json, os, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")


# From the logs
TRAIN_RE = re.compile(r"\[Train Epoch (\d+)\] Loss: ([0-9.]+), Dice: ([0-9.]+), IoU: ([0-9.]+)")
VAL_RE   = re.compile(r"\[Val Epoch (\d+)\] Loss: ([0-9.]+), Dice: ([0-9.]+), IoU: ([0-9.]+), Vol Error: ([0-9.]+)%")
BEST_RE  = re.compile(r"New best model saved! Dice: ([0-9.]+)")


# Parse through the logs
def parse_log(log_path: Path):
    train = {}
    val = {}
    best_epochs = []
    best_dices = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        last_val_epoch = None
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                ep = int(m.group(1))
                train[ep] = dict(loss=float(m.group(2)), dice=float(m.group(3)), iou=float(m.group(4)))
                continue
            m = VAL_RE.search(line)
            if m:
                ep = int(m.group(1))
                last_val_epoch = ep
                val[ep] = dict(loss=float(m.group(2)), dice=float(m.group(3)), iou=float(m.group(4)), vol=float(m.group(5)))
                continue
            m = BEST_RE.search(line)
            if m and last_val_epoch is not None:
                best_epochs.append(last_val_epoch)
                best_dices.append(float(m.group(1)))
    return train, val, best_epochs, best_dices


def to_arrays(d):
    eps = sorted(d.keys())
    arr = {k: np.array([d[e][k] for e in eps], dtype=float) for k in d[eps[0]].keys()} if eps else {}
    return np.array(eps, dtype=int), arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to run folder containing train.log and args.json")
    ap.add_argument("--out_dir", default=None, help="Where to save figures (default: run_dir/figures)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    log_path = run_dir / "train.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing {log_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, best_epochs, best_dices = parse_log(log_path)
    te, ta = to_arrays(train)
    ve, va = to_arrays(val)

    # If multiple folds exist later, call this script per-fold and then aggregate.
    # For now, single fold is plotted.

    def plot_metric(metric, ylabel, fname):
        plt.figure()
        if te.size and metric in ta:
            plt.plot(te, ta[metric], label="Train")
        if ve.size and metric in va:
            plt.plot(ve, va[metric], label="Val")
        # mark best epochs
        # for ep in best_epochs:
        #     plt.axvline(ep, linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()

    plot_metric("loss", "Loss", "curve_loss.png")
    plot_metric("dice", "Dice", "curve_dice.png")
    plot_metric("iou",  "IoU",  "curve_iou.png")

    # Print best epoch dice summary
    if best_epochs:
        best_ep = best_epochs[np.argmax(best_dices)]
        best_d  = max(best_dices)
        print(f"Best epoch (from log): {best_ep}  best dice={best_d:.4f}")
    else:
        print("No 'New best model saved' markers found in log.")

    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    main()