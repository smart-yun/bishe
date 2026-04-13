from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.runner import Runner
from mmengine.optim import build_optim_wrapper
from mmseg.models import build_segmentor
from mmseg.utils import register_all_modules


def setup_mmseg() -> None:
    register_all_modules(init_default_scope=True)


def load_cfg(config_path: str) -> Config:
    cfg = Config.fromfile(config_path)
    cfg.launcher = 'none'
    return cfg


def load_segmentor_from_checkpoint(
    config_path: str,
    checkpoint_path: str,
    device: str = 'cuda:0',
) -> Tuple[nn.Module, Config]:
    setup_mmseg()
    cfg = load_cfg(config_path)
    model = build_segmentor(cfg.model)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)

    if getattr(incompatible, 'missing_keys', None):
        print(f"[load] missing_keys: {len(incompatible.missing_keys)}")
    if getattr(incompatible, 'unexpected_keys', None):
        print(f"[load] unexpected_keys: {len(incompatible.unexpected_keys)}")

    model = revert_sync_batchnorm(model)
    if device.startswith('cuda') and torch.cuda.is_available():
        model = model.to(device)
    else:
        device = 'cpu'
        model = model.to('cpu')
    model.eval()
    return model, cfg


def tensor_forward(model: nn.Module, x: torch.Tensor) -> Any:
    return model(x, data_samples=None, mode='tensor')


def extract_logits_from_output(output: Any) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if torch.is_tensor(output):
        return output, []

    if isinstance(output, (list, tuple)):
        tensors = [x for x in output if torch.is_tensor(x)]
        if not tensors:
            raise RuntimeError('Model output list/tuple has no tensor logits.')
        return tensors[0], tensors[1:]

    if isinstance(output, dict):
        if 'logits' in output and torch.is_tensor(output['logits']):
            aux = output.get('aux_logits', [])
            if torch.is_tensor(aux):
                aux = [aux]
            elif isinstance(aux, (list, tuple)):
                aux = [x for x in aux if torch.is_tensor(x)]
            else:
                aux = []
            return output['logits'], aux

        tensors = [x for x in output.values() if torch.is_tensor(x)]
        if not tensors:
            raise RuntimeError('Model output dict has no tensor logits.')
        return tensors[0], tensors[1:]

    raise RuntimeError(f'Unsupported output type: {type(output)}')


def parse_losses(model: nn.Module, losses: Dict[str, Any]) -> torch.Tensor:
    if hasattr(model, 'parse_losses'):
        parsed_loss, _ = model.parse_losses(losses)
        return parsed_loss

    total = None
    for k, v in losses.items():
        if 'loss' not in k:
            continue
        if isinstance(v, torch.Tensor):
            total = v if total is None else total + v
        elif isinstance(v, (list, tuple)):
            cur = sum(x for x in v if isinstance(x, torch.Tensor))
            total = cur if total is None else total + cur
    if total is None:
        raise RuntimeError(f'Could not parse loss dict: keys={list(losses.keys())}')
    return total


def build_runner_for_cfg(cfg: Config, work_dir: str | None = None) -> Runner:
    cfg = cfg.copy()
    cfg.launcher = 'none'
    if work_dir is not None:
        cfg.work_dir = work_dir
    runner = Runner.from_cfg(cfg)
    return runner


def attach_pruned_model_to_runner(runner: Runner, model: nn.Module):
    if hasattr(runner, 'model') and runner.model is not None:
        del runner.model
    runner.model = model

    # Rebuild optimizer wrapper so it points to the pruned model parameters.
    runner.optim_wrapper = build_optim_wrapper(model, runner.cfg.optim_wrapper)
    return runner


def build_example_inputs(shape: Tuple[int, int], device: str) -> torch.Tensor:
    h, w = int(shape[0]), int(shape[1])
    x = torch.randn(1, 3, h, w)
    if device.startswith('cuda') and torch.cuda.is_available():
        x = x.to(device)
    return x


def count_ops_and_params(
    model: nn.Module,
    example_inputs: torch.Tensor,
    forward_fn=None,
) -> Tuple[int | None, int]:
    params = int(sum(p.numel() for p in model.parameters()))
    macs = None
    try:
        import torch_pruning as tp
        kwargs = {}
        if forward_fn is not None:
            kwargs['forward_fn'] = forward_fn
        macs, _ = tp.utils.count_ops_and_params(model, example_inputs, **kwargs)
        macs = int(macs)
    except Exception as e:
        print(f'[WARN] Failed to count MACs with torch_pruning: {e}')
    return macs, params


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class TensorModeWrapper(nn.Module):
    """Wrap a mmseg segmentor into a plain tensor forward model."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        return self.model(x, data_samples=None, mode='tensor')
