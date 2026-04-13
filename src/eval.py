from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmengine.runner import Runner

from utils_mmseg import load_cfg, setup_mmseg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline / pruned / finetuned SegFormer on RailSem19.",
        allow_abbrev=False,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work-dir", default="runs/tmp_eval")
    parser.add_argument("--output-json", default=None)

    parser.add_argument(
        "--config",
        default=None,
        help="Dataset / dataloader config path. Required together with --model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to full saved model object (.pth), e.g. pruned or finetuned full model.",
    )
    parser.add_argument(
        "--config-and-ckpt",
        nargs=2,
        metavar=("CONFIG", "CKPT"),
        default=None,
        help="Evaluate baseline / checkpoint-style model from config + checkpoint.",
    )

    args = parser.parse_args()

    has_model_branch = (args.config is not None) or (args.model is not None)
    has_ckpt_branch = args.config_and_ckpt is not None

    if has_model_branch and has_ckpt_branch:
        raise ValueError("Use either (--config + --model) or --config-and-ckpt, not both.")

    if not has_model_branch and not has_ckpt_branch:
        raise ValueError("You must provide either (--config + --model) or --config-and-ckpt.")

    if has_model_branch:
        if args.config is None or args.model is None:
            raise ValueError("Full-model evaluation requires both --config and --model.")

    return args


def save_metrics(metrics, output_json: str | None):
    if not output_json:
        return

    p = Path(output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON to: {p}")


def resolve_device(device_str: str) -> str:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return device_str
    return "cpu"


def main():
    args = parse_args()
    setup_mmseg()

    device = resolve_device(args.device)

    if args.config_and_ckpt is not None:
        # baseline / checkpoint branch
        config_path, ckpt_path = args.config_and_ckpt
        cfg = load_cfg(config_path)
        cfg.work_dir = args.work_dir
        cfg.resume = False
        cfg.load_from = ckpt_path
        cfg.launcher = "none"

        runner = Runner.from_cfg(cfg)
        metrics = runner.test()

    else:
        # full-model branch: config builds dataloaders/evaluator, model provides structure+weights
        cfg = load_cfg(args.config)
        cfg.work_dir = args.work_dir
        cfg.resume = False
        cfg.load_from = None
        cfg.launcher = "none"

        model = torch.load(args.model, map_location="cpu")
        if isinstance(model, dict):
            raise TypeError(
                "The file passed to --model is a checkpoint dict, not a full saved model object. "
                "For checkpoint-style weights, please use --config-and-ckpt CONFIG CKPT."
            )
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"--model expects a full torch.nn.Module, but got: {type(model)}")

        model = model.to(device)
        model.eval()

        runner = Runner.from_cfg(cfg)

        # Do not load checkpoint again; we already loaded the full model.
        runner._load_from = None
        runner._resume = False
        runner.model = model

        metrics = runner.test()

    print("----------------------------------------")
    print("Evaluation Results:")
    print(metrics)
    print("----------------------------------------")

    save_metrics(metrics, args.output_json)


if __name__ == "__main__":
    main()