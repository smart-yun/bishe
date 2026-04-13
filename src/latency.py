from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from utils_mmseg import (
    TensorModeWrapper,
    build_example_inputs,
    load_segmentor_from_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Measure latency for baseline or pruned SegFormer.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shape', type=int, nargs=2, default=[512, 512])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--repeat', type=int, default=300)
    parser.add_argument('--output-json', default=None)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', help='path to full saved model object (.pth)')
    group.add_argument('--config-and-ckpt', nargs=2, metavar=('CONFIG', 'CKPT'),
                       help='measure an unpruned model from config+checkpoint')

    return parser.parse_args()
#--config-and-ckpt config.py best_mIoU.pth

@torch.inference_mode()
def manual_latency(model: nn.Module, example_inputs: torch.Tensor, warmup: int, repeat: int):
    if example_inputs.device.type != 'cuda':
        # CPU fallback
        for _ in range(warmup):
            _ = model(example_inputs)

        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            _ = model(example_inputs)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        mean_ms = sum(times) / len(times)
        std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
        return mean_ms, std_ms

    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = model(example_inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        _ = model(example_inputs)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))

    mean_ms = float(sum(times) / len(times))
    std_ms = float((sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5)
    return mean_ms, std_ms


def main():
    args = parse_args()
    device = args.device if (args.device.startswith('cuda') and torch.cuda.is_available()) else 'cpu'

    if args.model is not None:
        model = torch.load(args.model, map_location='cpu')
    else:
        config_path, ckpt_path = args.config_and_ckpt
        model, _ = load_segmentor_from_checkpoint(config_path, ckpt_path, device=device)

    if device.startswith('cuda') and torch.cuda.is_available():
        model = model.to(device)
    else:
        model = model.to('cpu')

    wrapped = TensorModeWrapper(model).eval()
    h, w = int(args.shape[0]), int(args.shape[1])
    example_inputs = torch.randn(int(args.batch_size), 3, h, w)
    if device.startswith('cuda') and torch.cuda.is_available():
        example_inputs = example_inputs.to(device)

    mean_ms = std_ms = None

    try:
        import torch_pruning as tp
        mean_ms, std_ms = tp.utils.benchmark.measure_latency(
            wrapped,
            example_inputs=example_inputs,
            repeat=int(args.repeat),
        )
        mean_ms = float(mean_ms)
        std_ms = float(std_ms)
        backend = 'torch_pruning_official'
    except Exception as e:
        print(f'[WARN] tp.utils.benchmark.measure_latency failed, fallback to manual timer: {e}')
        mean_ms, std_ms = manual_latency(wrapped, example_inputs, warmup=int(args.warmup), repeat=int(args.repeat))
        backend = 'manual_fallback'

    fps = float(args.batch_size) / (mean_ms / 1000.0)

    print('----------------------------------------')
    print(f'Backend: {backend}')
    print(f'Latency: {mean_ms:.4f} ms')
    print(f'Std    : {std_ms:.4f} ms')
    print(f'FPS    : {fps:.2f}')
    print('----------------------------------------')

    if args.output_json:
        import json
        out = {
            'backend': backend,
            'latency_ms_mean': mean_ms,
            'latency_ms_std': std_ms,
            'fps': fps,
            'shape': list(args.shape),
            'batch_size': int(args.batch_size),
            'repeat': int(args.repeat),
            'warmup': int(args.warmup),
            'device': device,
        }
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
