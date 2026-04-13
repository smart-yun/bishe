import json
from pathlib import Path

import matplotlib.pyplot as plt


def moving_average(values, window=20):
    if window <= 1 or len(values) < window:
        return values[:]
    smoothed = []
    for i in range(len(values)):
        left = max(0, i - window + 1)
        smoothed.append(sum(values[left:i + 1]) / (i - left + 1))
    return smoothed


def plot_loss_and_miou(
    log_path: str,
    iters_per_epoch: int = 1700,
    save_path: str = "loss_miou_vs_epoch.png",
    smooth_loss_window: int = 20,
    show_best: bool = True,
):
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    train_steps = []
    train_epochs = []
    train_losses = []

    val_steps = []
    val_epochs = []
    val_mious = []

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "loss" in record and "step" in record:
                step = int(record["step"])
                loss = float(record["loss"])
                epoch = step / iters_per_epoch

                train_steps.append(step)
                train_epochs.append(epoch)
                train_losses.append(loss)

            if "mIoU" in record and "step" in record:
                step = int(record["step"])
                miou = float(record["mIoU"])
                epoch = step / iters_per_epoch

                val_steps.append(step)
                val_epochs.append(epoch)
                val_mious.append(miou)

    if not train_losses:
        raise ValueError("No training records with 'loss' were found in the log.")
    if not val_mious:
        raise ValueError("No validation records with 'mIoU' were found in the log.")

    smoothed_losses = moving_average(train_losses, window=smooth_loss_window)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(train_epochs, train_losses, alpha=0.25, linewidth=1, label="Loss (raw)")
    axes[0].plot(train_epochs, smoothed_losses, linewidth=2, label=f"Loss (MA{smooth_loss_window})")
    axes[0].set_title("Training Loss vs Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(val_epochs, val_mious, marker="o", linewidth=2, markersize=4, label="mIoU")

    if show_best:
        best_idx = max(range(len(val_mious)), key=lambda i: val_mious[i])
        best_epoch = val_epochs[best_idx]
        best_miou = val_mious[best_idx]
        best_step = val_steps[best_idx]

        axes[1].scatter([best_epoch], [best_miou], s=80, zorder=5, label="Best mIoU")
        axes[1].annotate(
            f"best={best_miou:.2f}\nepoch={best_epoch:.1f}\nstep={best_step}",
            xy=(best_epoch, best_miou),
            xytext=(best_epoch + 2, best_miou - 1.5),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10,
        )

    axes[1].set_title("Validation mIoU vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU (%)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    log_file = "runs/rs19/segformer_b1_512x512_100ep_rtx4090/20260411_135344/vis_data/20260411_135344.json"

    plot_loss_and_miou(
        log_path=log_file,
        iters_per_epoch=1700,
        save_path="loss_miou_vs_epoch.png",
        smooth_loss_window=20,
        show_best=True,
    )