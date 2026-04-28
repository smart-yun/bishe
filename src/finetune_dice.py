from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from mmengine.optim import build_optim_wrapper
from mmengine.runner import Runner
from mmseg.registry import MODELS

from utils_mmseg import load_cfg, setup_mmseg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a structurally pruned SegFormer with CE + Dice loss."
    )
    parser.add_argument("--config", required=True, help="baseline mmseg config path")
    parser.add_argument("--pruned-model", required=True, help="path to full pruned model object (.pth)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ce-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=float, default=0.5)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_str: str) -> str:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return device_str
    return "cpu"


def disable_pretrained_init_cfg(node):
    if node is None:
        return

    try:
        if "pretrained" in node:
            node["pretrained"] = None
    except Exception:
        pass

    try:
        if "init_cfg" in node:
            node["init_cfg"] = None
    except Exception:
        pass

    try:
        items = list(node.items())
    except Exception:
        return

    for _, value in items:
        if isinstance(value, (dict, list, tuple)):
            disable_pretrained_init_cfg(value)


def load_checkpoint_state_dict(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]

    if isinstance(ckpt, dict):
        return ckpt

    raise TypeError(f"Unexpected checkpoint type from {ckpt_path}: {type(ckpt)}")


def set_ce_dice_loss(model: torch.nn.Module, ce_weight: float, dice_weight: float) -> None:
    ce_loss = MODELS.build(
        dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_name="loss_ce",
            loss_weight=ce_weight,
        )
    )

    dice_loss = MODELS.build(
        dict(
            type="DiceLoss",
            use_sigmoid=False,
            activate=True,
            naive_dice=True,
            eps=1e-3,
            loss_name="loss_dice",
            loss_weight=dice_weight,
        )
    )

    if not hasattr(model, "decode_head"):
        raise AttributeError("The loaded model does not have decode_head.")

    model.decode_head.loss_decode = torch.nn.ModuleList([ce_loss, dice_loss])

    print("----------------------------------------")
    print("Loss has been replaced by CE + Dice:")
    print(f"CE weight: {ce_weight}")
    print(f"Dice weight: {dice_weight}")
    print("----------------------------------------")


def save_full_models(
    work_dir: Path,
    trained_model: torch.nn.Module,
    pruned_model_template: torch.nn.Module,
):
    work_dir.mkdir(parents=True, exist_ok=True)

    latest_full_model = work_dir / "latest_full_model.pth"
    torch.save(trained_model, latest_full_model)
    print(f"Saved latest full model to: {latest_full_model}")

    best_candidates = sorted(work_dir.glob("best_mIoU*.pth"))
    if not best_candidates:
        print("[WARN] No best_mIoU*.pth found. Skip saving best_full_model.pth")
        return

    best_ckpt = best_candidates[-1]
    state_dict = load_checkpoint_state_dict(best_ckpt)

    best_model = copy.deepcopy(pruned_model_template).cpu()
    set_ce_dice_loss(best_model, ce_weight=1.0, dice_weight=0.5)
    best_model.load_state_dict(state_dict, strict=True)

    best_full_model = work_dir / "best_full_model.pth"
    torch.save(best_model, best_full_model)

    print(f"Saved best full model to: {best_full_model}")
    print(f"Matched best checkpoint: {best_ckpt.name}")


def main():
    args = parse_args()
    setup_mmseg()

    cfg = load_cfg(args.config)
    cfg.work_dir = args.work_dir
    cfg.resume = False
    cfg.load_from = None

    disable_pretrained_init_cfg(cfg.model)

    iters_per_epoch = int(getattr(cfg, "iters_per_epoch", 0))
    if iters_per_epoch <= 0:
        raise RuntimeError(
            "The config must expose `iters_per_epoch` for epoch-equivalent finetuning."
        )

    ft_max_iters = int(args.finetune_epochs) * iters_per_epoch
    cfg.train_cfg.max_iters = ft_max_iters
    cfg.train_cfg.val_interval = (
        int(args.val_interval) if args.val_interval is not None else iters_per_epoch
    )

    if "optimizer" in cfg.optim_wrapper:
        cfg.optim_wrapper.optimizer.lr = float(args.lr)
        cfg.optim_wrapper.optimizer.weight_decay = float(args.weight_decay)

    if "checkpoint" in cfg.default_hooks:
        cfg.default_hooks.checkpoint.by_epoch = False
        cfg.default_hooks.checkpoint.interval = (
            int(args.save_interval) if args.save_interval is not None else iters_per_epoch
        )
        cfg.default_hooks.checkpoint.save_best = "mIoU"

    if "logger" in cfg.default_hooks:
        cfg.default_hooks.logger.interval = max(50, iters_per_epoch // 10)
        cfg.default_hooks.logger.log_metric_by_epoch = False

    runner = Runner.from_cfg(cfg)

    device = resolve_device(args.device)

    model = torch.load(args.pruned_model, map_location="cpu")
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"--pruned-model must be a full torch.nn.Module, got {type(model)}")

    model = model.to(device)

    set_ce_dice_loss(
        model=model,
        ce_weight=args.ce_loss_weight,
        dice_weight=args.dice_loss_weight,
    )

    pruned_model_template = copy.deepcopy(model).cpu()

    runner.model = model
    runner.optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)

    print("----------------------------------------")
    print("Finetune with CE + Dice setup:")
    print(f"Pruned model: {args.pruned_model}")
    print(f"Work dir: {cfg.work_dir}")
    print(f"Finetune epochs: {args.finetune_epochs}")
    print(f"Max iters: {cfg.train_cfg.max_iters}")
    print(f"Val interval: {cfg.train_cfg.val_interval}")
    print(f"LR: {cfg.optim_wrapper.optimizer.lr}")
    print(f"Weight decay: {cfg.optim_wrapper.optimizer.weight_decay}")
    print("----------------------------------------")

    runner.train()

    save_full_models(
        work_dir=Path(cfg.work_dir),
        trained_model=runner.model.cpu(),
        pruned_model_template=pruned_model_template,
    )


if __name__ == "__main__":
    main()