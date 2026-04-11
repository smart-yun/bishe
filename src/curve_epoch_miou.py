# -*- coding: utf-8 -*-
"""Plot mIoU/loss curves from MMEngine line-json log.

python src/curve_epoch_miou.py \
--log-json runs/rs19/segformer_b0_512x512_100ep_rtx4090_from_best/20260409_163249/vis_data/20260409_163249.json \
--out-png results/curves/miou_loss_curve_100ep_from_best_iter.png \
--out-epoch-png results/curves/miou_loss_curve_100ep_from_best_epoch.png \
--out-epoch-loss-png results/curves/loss_curve_100ep_from_best_epoch.png \
--out-summary-json results/curves/miou_loss_curve_100ep_from_best_summary.json \
--iters-per-epoch 1700 && ls -l results/curves/miou_loss_curve_100ep_from_best_iter.png results/curves/miou_loss_curve_100ep_from_best_epoch.png results/curves/loss_curve_100ep_from_best_epoch.png results/curves/miou_loss_curve_100ep_from_best_summary.json



"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def moving_avg(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) < window:
        return values
    out: List[float] = []
    csum = 0.0
    q: List[float] = []
    for v in values:
        q.append(v)
        csum += v
        if len(q) > window:
            csum -= q.pop(0)
        out.append(csum / len(q))
    return out


def _step_to_epoch(step: int, iters_per_epoch: int | None) -> int | None:
    if iters_per_epoch is None or iters_per_epoch <= 0:
        return None
    return (int(step) - 1) // iters_per_epoch + 1


def parse_mmengine_json(
    log_json: Path,
    iters_per_epoch: int | None = None,
) -> Tuple[List[int], List[float], List[int], List[float], List[int], List[float], List[int], List[float]]:
    train_steps: List[int] = []
    train_losses: List[float] = []
    val_map: Dict[int, float] = {}
    val_epoch_map: Dict[int, float] = {}
    train_epoch_losses: Dict[int, List[float]] = {}

    with log_json.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            step = obj.get("step", obj.get("iter", None))
            if step is None:
                continue
            step = int(step)

            if "loss" in obj:
                loss = float(obj["loss"])
                train_steps.append(step)
                train_losses.append(loss)
                e = _step_to_epoch(step, iters_per_epoch)
                if e is not None:
                    train_epoch_losses.setdefault(e, []).append(loss)

            if "mIoU" in obj:
                miou = float(obj["mIoU"])
                val_map[step] = miou
                epoch = obj.get("epoch", None)
                if epoch is not None:
                    val_epoch_map[int(epoch)] = miou
                else:
                    e = _step_to_epoch(step, iters_per_epoch)
                    if e is not None:
                        val_epoch_map[e] = miou

    val_steps = sorted(val_map.keys())
    val_mious = [val_map[s] for s in val_steps]
    val_epochs = sorted(val_epoch_map.keys())
    val_mious_by_epoch = [val_epoch_map[e] for e in val_epochs]

    train_epochs = sorted(train_epoch_losses.keys())
    train_loss_by_epoch = [sum(train_epoch_losses[e]) / max(1, len(train_epoch_losses[e])) for e in train_epochs]

    return train_steps, train_losses, val_steps, val_mious, val_epochs, val_mious_by_epoch, train_epochs, train_loss_by_epoch


def nearest_prev_train_loss(train_steps, train_losses, val_step):
    idx = None
    for i, s in enumerate(train_steps):
        if s > val_step:
            break
        idx = i
    return train_losses[idx] if idx is not None else None


def detect_overfit(train_steps, train_losses, val_steps, val_mious, miou_drop_threshold=0.5, loss_drop_threshold=0.05):
    result = {"overfit_found": False, "overfit_step": None, "overfit_miou_drop": None, "overfit_loss_drop": None}
    if len(val_steps) < 2:
        return result
    for i in range(1, len(val_steps)):
        miou_drop = val_mious[i - 1] - val_mious[i]
        loss_prev = nearest_prev_train_loss(train_steps, train_losses, val_steps[i - 1])
        loss_now = nearest_prev_train_loss(train_steps, train_losses, val_steps[i])
        if loss_prev is not None and loss_now is not None:
            loss_drop = loss_prev - loss_now
            if miou_drop > miou_drop_threshold and loss_drop > loss_drop_threshold:
                result.update({"overfit_found": True, "overfit_step": val_steps[i], "overfit_miou_drop": miou_drop, "overfit_loss_drop": loss_drop})
                break
    return result


def detect_overfit_critical_step(train_steps, train_losses, val_steps, val_mious, miou_drop_threshold=0.5, loss_drop_threshold=0.05, consecutive_val_points=3):
    result = {"critical_found": False, "critical_step": None}
    if len(val_steps) < consecutive_val_points:
        return result
    for i in range(consecutive_val_points, len(val_steps)):
        overfit = True
        for j in range(i - consecutive_val_points + 1, i + 1):
            miou_drop = val_mious[j - 1] - val_mious[j]
            loss_prev = nearest_prev_train_loss(train_steps, train_losses, val_steps[j - 1])
            loss_now = nearest_prev_train_loss(train_steps, train_losses, val_steps[j])
            if not (miou_drop > miou_drop_threshold and loss_prev is not None and loss_now is not None and (loss_prev - loss_now) > loss_drop_threshold):
                overfit = False
                break
        if overfit:
            result.update({"critical_found": True, "critical_step": val_steps[i]})
            break
    return result


def plot_curve(
    train_steps: List[int],
    train_losses: List[float],
    val_steps: List[int],
    val_mious: List[float],
    out_png: Path,
    smooth_window: int,
    critical_step: int | None = None,
    val_epochs: List[int] | None = None,
    val_mious_by_epoch: List[float] | None = None,
    train_epochs: List[int] | None = None,
    train_loss_by_epoch: List[float] | None = None,
    out_epoch_png: Path | None = None,
    out_epoch_loss_png: Path | None = None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 5.6), dpi=150)
    ax1.plot(val_steps, val_mious, marker="o", linewidth=1.8, label="val mIoU (iter)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("mIoU (%)")
    ax1.grid(True, linestyle="--", alpha=0.35)

    if val_mious:
        best_idx = max(range(len(val_mious)), key=lambda i: val_mious[i])
        bx, by = val_steps[best_idx], val_mious[best_idx]
        ax1.scatter([bx], [by], s=50)
        ax1.annotate(f"best={by:.2f}@{bx}", (bx, by), xytext=(6, 6), textcoords="offset points")

    if critical_step is not None:
        ax1.axvline(critical_step, linestyle="--", linewidth=1.4, alpha=0.8, label=f"overfit critical@{critical_step}")

    ax2 = ax1.twinx()
    if train_losses:
        smooth_loss = moving_avg(train_losses, smooth_window)
        ax2.plot(train_steps, smooth_loss, linewidth=1.0, alpha=0.65, label=f"train loss (ma{smooth_window})")
    ax2.set_ylabel("Train Loss")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("mIoU / Train Loss vs Iteration")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)

    if out_epoch_png and val_epochs and val_mious_by_epoch:
        fig2, ax3 = plt.subplots(figsize=(10, 5.6), dpi=150)
        ax3.plot(val_epochs, val_mious_by_epoch, marker="o", linewidth=2.0, color="tab:blue", label="val mIoU (epoch)")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("mIoU (%)", color="tab:blue")
        ax3.tick_params(axis="y", labelcolor="tab:blue")
        ax3.grid(True, linestyle="--", alpha=0.35)

        best_idx = max(range(len(val_mious_by_epoch)), key=lambda i: val_mious_by_epoch[i])
        bx, by = val_epochs[best_idx], val_mious_by_epoch[best_idx]
        ax3.scatter([bx], [by], s=50, color="tab:blue")
        ax3.annotate(f"best={by:.2f}@{bx}", (bx, by), xytext=(6, 6), textcoords="offset points")

        ax4 = ax3.twinx()
        if train_epochs and train_loss_by_epoch:
            ax4.plot(train_epochs, train_loss_by_epoch, marker="s", linewidth=1.6, alpha=0.85, color="tab:red", label="train loss (epoch avg)")
        ax4.set_ylabel("Train Loss", color="tab:red")
        ax4.tick_params(axis="y", labelcolor="tab:red")

        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax3.legend(lines3 + lines4, labels3 + labels4, loc="best")

        plt.title("mIoU / Train Loss vs Epoch")
        plt.tight_layout()
        out_epoch_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_epoch_png)
        plt.close(fig2)

        if out_epoch_loss_png is None:
            out_epoch_loss_png = out_epoch_png.with_name(out_epoch_png.stem + "_loss.png")

        if train_epochs and train_loss_by_epoch and out_epoch_loss_png:
            fig3, ax5 = plt.subplots(figsize=(10, 5.6), dpi=150)
            ax5.plot(train_epochs, train_loss_by_epoch, marker="o", linewidth=2.0, color="tab:red", label="train loss (epoch avg)")
            ax5.set_xlabel("Epoch")
            ax5.set_ylabel("Train Loss")
            ax5.grid(True, linestyle="--", alpha=0.35)
            ax5.legend(loc="best")
            plt.title("Train Loss vs Epoch")
            plt.tight_layout()
            out_epoch_loss_png.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_epoch_loss_png)
            plt.close(fig3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mIoU/loss curves from MMEngine log json")
    parser.add_argument("--log-json", required=True, help="Path to line-by-line JSON log")
    parser.add_argument("--out-png", required=True, help="Output curve image path (iter)")
    parser.add_argument("--out-epoch-png", required=False, help="Output mIoU/loss epoch curve image path")
    parser.add_argument("--out-epoch-loss-png", required=False, help="Optional output loss-epoch-only curve image path")
    parser.add_argument("--out-summary-json", required=True, help="Output analysis json path")
    parser.add_argument("--smooth-window", type=int, default=50, help="Moving-average window for train loss")
    parser.add_argument("--miou-drop-threshold", type=float, default=0.5, help="mIoU drop threshold for overfit warning")
    parser.add_argument("--loss-drop-threshold", type=float, default=0.05, help="Train-loss drop threshold for overfit warning")
    parser.add_argument("--consecutive-val-points", type=int, default=3, help="Consecutive val points required to confirm critical overfit")
    parser.add_argument("--iters-per-epoch", type=int, default=None, help="Iters per epoch (if log has no epoch field)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_json = Path(args.log_json)
    out_png = Path(args.out_png)
    out_summary = Path(args.out_summary_json)
    out_epoch_png = Path(args.out_epoch_png) if args.out_epoch_png else None
    out_epoch_loss_png = Path(args.out_epoch_loss_png) if args.out_epoch_loss_png else None

    train_steps, train_losses, val_steps, val_mious, val_epochs, val_mious_by_epoch, train_epochs, train_loss_by_epoch = parse_mmengine_json(
        log_json, iters_per_epoch=args.iters_per_epoch
    )

    if not val_steps:
        raise RuntimeError("No mIoU found in log. Please ensure this is a training log with validation metrics.")

    summary = detect_overfit(
        train_steps=train_steps,
        train_losses=train_losses,
        val_steps=val_steps,
        val_mious=val_mious,
        miou_drop_threshold=args.miou_drop_threshold,
        loss_drop_threshold=args.loss_drop_threshold,
    )

    critical = detect_overfit_critical_step(
        train_steps=train_steps,
        train_losses=train_losses,
        val_steps=val_steps,
        val_mious=val_mious,
        miou_drop_threshold=args.miou_drop_threshold,
        loss_drop_threshold=args.loss_drop_threshold,
        consecutive_val_points=args.consecutive_val_points,
    )

    plot_curve(
        train_steps=train_steps,
        train_losses=train_losses,
        val_steps=val_steps,
        val_mious=val_mious,
        out_png=out_png,
        smooth_window=args.smooth_window,
        critical_step=critical["critical_step"] if critical["critical_found"] else None,
        val_epochs=val_epochs,
        val_mious_by_epoch=val_mious_by_epoch,
        train_epochs=train_epochs,
        train_loss_by_epoch=train_loss_by_epoch,
        out_epoch_png=out_epoch_png,
        out_epoch_loss_png=out_epoch_loss_png,
    )

    summary["num_val_points"] = len(val_steps)
    summary["first_step"] = val_steps[0]
    summary["num_epoch_points"] = len(val_epochs)
    summary["first_epoch"] = val_epochs[0] if val_epochs else None
    summary["min_train_loss"] = min(train_losses) if train_losses else None
    summary["max_train_loss"] = max(train_losses) if train_losses else None
    summary["last_train_loss"] = train_losses[-1] if train_losses else None
    summary["min_epoch_loss"] = min(train_loss_by_epoch) if train_loss_by_epoch else None
    summary["last_epoch_loss"] = train_loss_by_epoch[-1] if train_loss_by_epoch else None

    if val_mious:
        best_idx = max(range(len(val_mious)), key=lambda i: val_mious[i])
        summary["best_mIoU"] = val_mious[best_idx]
        summary["best_mIoU_step"] = val_steps[best_idx]
    if val_mious_by_epoch:
        best_eidx = max(range(len(val_mious_by_epoch)), key=lambda i: val_mious_by_epoch[i])
        summary["best_mIoU_epoch"] = val_epochs[best_eidx]

    summary.update(critical)

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] curve saved: {out_png}")
    if out_epoch_png:
        print(f"[OK] epoch curve saved: {out_epoch_png}")
        if out_epoch_loss_png:
            print(f"[OK] epoch loss curve saved: {out_epoch_loss_png}")
        else:
            print(f"[OK] epoch loss curve saved: {out_epoch_png.with_name(out_epoch_png.stem + '_loss.png')}")
    print(f"[OK] summary saved: {out_summary}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
