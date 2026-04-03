# -*- coding: utf-8 -*-
import os
import sys


def _bootstrap_runtime() -> None:
    """Ensure the script runs with an interpreter that has mmcv/mmseg installed."""
    try:
        import mmcv  # noqa: F401
        return
    except ModuleNotFoundError as exc:
        if exc.name != 'mmcv':
            raise

    candidates = []
    env_python = os.environ.get('BISHE_PYTHON')
    if env_python:
        candidates.append(env_python)

    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        candidates.extend([
            os.path.join(conda_prefix, 'envs', 'railseg', 'bin', 'python'),
            os.path.join(conda_prefix, 'envs', 'railseg2', 'bin', 'python'),
        ])

    candidates.extend([
        '/home/lcy/miniconda3/envs/railseg/bin/python',
        '/home/lcy/miniconda3/envs/railseg2/bin/python',
    ])

    current_python = os.path.realpath(sys.executable)
    for candidate in candidates:
        if candidate and os.path.exists(candidate) and os.path.realpath(candidate) != current_python:
            os.execv(candidate, [candidate, *sys.argv])

    raise ModuleNotFoundError(
        'No module named \'mmcv\'. Please run this script with the railseg environment or set BISHE_PYTHON to the correct Python interpreter.'
    )


_bootstrap_runtime()

import torch
import torch_pruning as tp

from mmengine.config import Config
from mmseg.models import build_segmentor
from mmengine.model import revert_sync_batchnorm
try:
    # mmsegmentation >= 1.x
    from mmseg.models.backbones.mit import MixFFN
except ImportError:
    # ?????? mmsegmentation
    from mmseg.models.backbones.mix_transformer import MixFFN

def get_model_from_config(config_path, checkpoint_path):
    cfg = Config.fromfile(config_path)
    model = build_segmentor(cfg.model)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    model = revert_sync_batchnorm(model)
    return model, cfg

def main():
    # --- 1. ???? ---
    config_path = 'configs/railsem19/segformer_b0_rs19_512x512_40000it.py'
    checkpoint_path = 'runs/rs19/segformer_b0_512x512_40000it/best_mIoU_iter_40000.pth'
    pruning_ratio = 0.2
    
    print(f"Loading model from config '{config_path}'...")
    model, cfg = get_model_from_config(config_path, checkpoint_path)
    
    # --- 2. ????????? ---
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1, 3, 512, 512))
    
    targets_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, MixFFN):
            target_conv = module.layers[0]
            targets_to_prune.append(target_conv)
    
    print(f"Found {len(targets_to_prune)} target layers to prune.")

    print(f"Starting global pruning with ratio: {pruning_ratio*100}%...")
    for i, target_layer in enumerate(targets_to_prune):
        # ????L1?????????????????????
        weight = target_layer.weight.detach()
        l1 = weight.abs().flatten(1).sum(1)
        n_prune = int(target_layer.out_channels * pruning_ratio)
        idxs = torch.argsort(l1)[:n_prune].tolist()

        # ????????
        group = DG.get_pruning_group(
            target_layer,
            tp.prune_conv_out_channels,
            idxs=idxs
        )
        
        # ??ùù??
        if hasattr(group, 'is_pruned') and group.is_pruned:
            print(f"Warning: Target {i+1} has been pruned, skipping.")
        else:
            group.prune()

    print("Global pruning finished.")

    print("\n--- Pruned Model Structure ---")
    print(model)
    
    try:
        print("\nTesting forward pass...")
        with torch.no_grad():
            test_input = torch.randn(1, 3, 512, 512)
            output = model(test_input)
        print(f"Forward pass successful! Output shape: {output[0].shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")

    pruned_checkpoint_path = f'checkpoints/globally_pruned_ffn_{int(pruning_ratio*100)}p.pth'
    os.makedirs(os.path.dirname(pruned_checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), pruned_checkpoint_path)
    print(f"\nPruned model saved to: {pruned_checkpoint_path}")

if __name__ == '__main__':
    main()
