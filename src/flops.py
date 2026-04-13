from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from utils_mmseg import TensorModeWrapper, load_segmentor_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Measure Params / MACs / FLOPs for baseline or pruned SegFormer.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shape", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-json", default=None)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Path to full saved model object (.pth)")
    group.add_argument(
        "--config-and-ckpt",
        nargs=2,
        metavar=("CONFIG", "CKPT"),
        help="Measure an unpruned model from config + checkpoint",
    )

    return parser.parse_args()


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def try_fvcore(model: nn.Module, example_inputs: torch.Tensor) -> Tuple[str, int]:
    from fvcore.nn import FlopCountAnalysis

    analysis = FlopCountAnalysis(model, example_inputs)
    flops = int(analysis.total())
    return "fvcore", flops


def try_ptflops(model: nn.Module, shape: Tuple[int, int], batch_size: int) -> Tuple[str, int]:
    from ptflops import get_model_complexity_info

    h, w = int(shape[0]), int(shape[1])

    def input_constructor(_: Tuple[int, ...]) -> Dict[str, Any]:
        x = torch.randn(batch_size, 3, h, w)
        device = next(model.parameters()).device
        x = x.to(device)
        return {"x": x}

    macs_str, _ = get_model_complexity_info(
        model,
        (3, h, w),
        input_constructor=input_constructor,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )

    macs = int(macs_str)
    return "ptflops", macs


def try_thop(model: nn.Module, example_inputs: torch.Tensor) -> Tuple[str, int]:
    from thop import profile

    macs, _ = profile(model, inputs=(example_inputs,), verbose=False)
    macs = int(macs)
    return "thop", macs


def measure_complexity(model: nn.Module, example_inputs: torch.Tensor, shape: Tuple[int, int], batch_size: int):
    errors = []

    try:
        backend, flops = try_fvcore(model, example_inputs)
        macs = flops // 2
        return {
            "backend": backend,
            "params": count_params(model),
            "macs": macs,
            "flops": flops,
            "errors": errors,
        }
    except Exception as e:
        errors.append(f"fvcore failed: {e}")

    try:
        backend, macs = try_ptflops(model, shape, batch_size)
        flops = macs * 2
        return {
            "backend": backend,
            "params": count_params(model),
            "macs": macs,
            "flops": flops,
            "errors": errors,
        }
    except Exception as e:
        errors.append(f"ptflops failed: {e}")

    try:
        backend, macs = try_thop(model, example_inputs)
        flops = macs * 2
        return {
            "backend": backend,
            "params": count_params(model),
            "macs": macs,
            "flops": flops,
            "errors": errors,
        }
    except Exception as e:
        errors.append(f"thop failed: {e}")

    return {
        "backend": None,
        "params": count_params(model),
        "macs": None,
        "flops": None,
        "errors": errors,
    }


def main():
    args = parse_args()
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    if args.model is not None:
        model = torch.load(args.model, map_location="cpu")
        if isinstance(model, dict):
            raise TypeError(
                "The file passed to --model is a checkpoint dict, not a full saved model object. "
                "For baseline checkpoints, please use --config-and-ckpt CONFIG CKPT. "
                "Use --model only for full models saved by torch.save(model, path)."
            )
    else:
        config_path, ckpt_path = args.config_and_ckpt
        model, _ = load_segmentor_from_checkpoint(config_path, ckpt_path, device=device)

    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.to(device)
    else:
        model = model.to("cpu")
        device = "cpu"

    wrapped = TensorModeWrapper(model).eval()

    h, w = int(args.shape[0]), int(args.shape[1])
    example_inputs = torch.randn(int(args.batch_size), 3, h, w)
    if device.startswith("cuda") and torch.cuda.is_available():
        example_inputs = example_inputs.to(device)

    result = measure_complexity(
        model=wrapped,
        example_inputs=example_inputs,
        shape=(h, w),
        batch_size=int(args.batch_size),
    )

    params = result["params"]
    macs = result["macs"]
    flops = result["flops"]
    backend = result["backend"]

    print("----------------------------------------")
    print(f"Backend: {backend}")
    print(f"Params : {params:,} ({params / 1e6:.4f} M)")
    if macs is not None:
        print(f"MACs   : {macs:,} ({macs / 1e9:.4f} G)")
    else:
        print("MACs   : None")
    if flops is not None:
        print(f"FLOPs  : {flops:,} ({flops / 1e9:.4f} G)")
    else:
        print("FLOPs  : None")
    print("----------------------------------------")

    if result["errors"]:
        print("Fallback log:")
        for err in result["errors"]:
            print(f"- {err}")

    if args.output_json:
        out = {
            "backend": backend,
            "params": params,
            "params_m": params / 1e6,
            "macs": macs,
            "macs_g": None if macs is None else macs / 1e9,
            "flops": flops,
            "flops_g": None if flops is None else flops / 1e9,
            "shape": list(args.shape),
            "batch_size": int(args.batch_size),
            "device": device,
            "errors": result["errors"],
        }
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON to: {p}")


if __name__ == "__main__":
    main()