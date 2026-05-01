#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Suitable for SSH/server environments
import matplotlib.pyplot as plt


# =========================
# Only modify these settings
# =========================
LOG_FILE = "runs/rs19/b1_p50_ft30_kd_logit_T4_w01/20260427_104810/vis_data/20260427_104810.json"
OUT_FILE = None          # None means saving to the same directory as the log file
SMOOTH_WINDOW = 5        # Set to 1 to disable smoothing


def moving_average(values, window=5):
    """Apply simple moving average smoothing."""
    if window <= 1 or len(values) == 0:
        return values

    smoothed = []
    for i in range(len(values)):
        left = max(0, i - window + 1)
        part = values[left:i + 1]
        smoothed.append(sum(part) / len(part))
    return smoothed


def read_mmengine_json_log(log_file):
    """
    Read MMEngine / MMSegmentation JSON Lines logs.

    Training records usually contain:
        step, loss, decode.loss_ce, loss_kd_logit

    Validation records usually contain:
        step, mIoU, aAcc, mAcc
    """
    log_file = Path(log_file)

    train_steps = []
    train_losses = []

    val_steps = []
    val_mious = []

    ce_losses = []
    kd_losses = []

    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            step = record.get("step", record.get("iter", None))
            if step is None:
                continue

            # Training loss
            if "loss" in record:
                train_steps.append(step)
                train_losses.append(record["loss"])

                ce_losses.append(record.get("decode.loss_ce", None))
                kd_losses.append(record.get("loss_kd_logit", None))

            # Validation mIoU
            if "mIoU" in record:
                val_steps.append(step)
                val_mious.append(record["mIoU"])

    return {
        "train_steps": train_steps,
        "train_losses": train_losses,
        "ce_losses": ce_losses,
        "kd_losses": kd_losses,
        "val_steps": val_steps,
        "val_mious": val_mious,
    }


def plot_miou_loss(
    log_file,
    out_file=None,
    smooth_window=5,
    show_ce_loss=False,
    show_kd_loss=False,
):
    """
    Plot mIoU and loss curves from an MMEngine JSON log.

    Args:
        log_file: Path to the JSON log file.
        out_file: Output image path. If None, save beside the log file.
        smooth_window: Moving-average window for loss smoothing.
        show_ce_loss: Whether to also plot decode.loss_ce.
        show_kd_loss: Whether to also plot loss_kd_logit.
    """
    log_file = Path(log_file)
    data = read_mmengine_json_log(log_file)

    train_steps = data["train_steps"]
    train_losses = data["train_losses"]
    ce_losses = data["ce_losses"]
    kd_losses = data["kd_losses"]
    val_steps = data["val_steps"]
    val_mious = data["val_mious"]

    if len(train_steps) == 0:
        raise ValueError("No training loss was found. Please check whether the log contains the 'loss' field.")

    if len(val_steps) == 0:
        raise ValueError("No validation mIoU was found. Please check whether the log contains the 'mIoU' field.")

    if out_file is None:
        out_file = log_file.parent / "miou_loss_curve.png"
    else:
        out_file = Path(out_file)

    loss_to_plot = moving_average(train_losses, smooth_window)

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # Left axis: training loss
    ax1.plot(
        train_steps,
        loss_to_plot,
        linewidth=1.8,
        label=f"Train Loss, smooth={smooth_window}",
    )
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Optional: CE loss
    if show_ce_loss and any(v is not None for v in ce_losses):
        ce_steps = [s for s, v in zip(train_steps, ce_losses) if v is not None]
        ce_values = [v for v in ce_losses if v is not None]
        ce_values = moving_average(ce_values, smooth_window)

        ax1.plot(
            ce_steps,
            ce_values,
            linewidth=1.2,
            linestyle="--",
            label="CE Loss",
        )

    # Optional: KD loss
    if show_kd_loss and any(v is not None for v in kd_losses):
        kd_steps = [s for s, v in zip(train_steps, kd_losses) if v is not None]
        kd_values = [v for v in kd_losses if v is not None]
        kd_values = moving_average(kd_values, smooth_window)

        ax1.plot(
            kd_steps,
            kd_values,
            linewidth=1.2,
            linestyle=":",
            label="KD Logit Loss",
        )

    # Right axis: validation mIoU
    ax2 = ax1.twinx()
    ax2.plot(
        val_steps,
        val_mious,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="Val mIoU",
    )
    ax2.set_ylabel("mIoU (%)")

    # Mark best mIoU
    best_idx = val_mious.index(max(val_mious))
    best_step = val_steps[best_idx]
    best_miou = val_mious[best_idx]

    ax2.scatter(
        [best_step],
        [best_miou],
        s=120,
        marker="*",
        zorder=5,
        label=f"Best mIoU = {best_miou:.2f}",
    )

    ax2.annotate(
        f"Best mIoU: {best_miou:.2f}\nStep: {best_step}",
        xy=(best_step, best_miou),
        xytext=(20, 15),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.25),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
    )

    # Merge legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.title(f"mIoU and Loss Curve\n{log_file.name}")
    plt.tight_layout()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure to: {out_file}")
    print(f"[INFO] Training loss points: {len(train_steps)}")
    print(f"[INFO] Validation mIoU points: {len(val_steps)}")
    print(f"[INFO] Best mIoU: {best_miou:.2f} at step {best_step}")


if __name__ == "__main__":
    plot_miou_loss(
        log_file=LOG_FILE,
        out_file=OUT_FILE,
        smooth_window=SMOOTH_WINDOW,
        show_ce_loss=False,
        show_kd_loss=False,
    )