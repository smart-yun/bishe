from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch_pruning as tp

from utils_mmseg import (
    build_example_inputs,
    build_runner_for_cfg,
    count_ops_and_params,
    extract_logits_from_output,
    load_segmentor_from_checkpoint,
    parse_losses,
    save_json,
    tensor_forward,
)


def parse_args():
    parser = argparse.ArgumentParser(description='SegFormer-B1 pruning in Torch-Pruning official-example style.')

    parser.add_argument('--config', required=True, help='mmseg config path')
    parser.add_argument('--checkpoint', required=True, help='baseline checkpoint path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shape', type=int, nargs=2, default=[512, 512])
    

    parser.add_argument('--output-dir', default='output/segformer_pruned')
    parser.add_argument('--save-model-name', default='model_pruned.pth')

    parser.add_argument('--importance', choices=['group_magnitude', 'magnitude', 'taylor'], default='group_magnitude')
    parser.add_argument('--mode', choices=['mlp_bottleneck', 'uniform_linear'], default='mlp_bottleneck')

    parser.add_argument('--pruning-ratio', type=float, default=0.15)
    parser.add_argument('--iterative-steps', type=int, default=5)
    parser.add_argument('--max-pruning-ratio', type=float, default=0.20)
    parser.add_argument('--round-to', type=int, default=8)

    parser.add_argument('--global-pruning', action='store_true')
    parser.add_argument('--isomorphic', action='store_true')

    parser.add_argument('--prune-stages', type=int, nargs='*', default=[2, 3, 4],
                        help='1-based stage ids to prune, e.g. 2 3 4')
    parser.add_argument('--protect-keywords', nargs='*',
                        default=['decode_head', 'auxiliary_head'],
                        help='Always protect these module-name keywords')

    parser.add_argument('--taylor-batches', type=int, default=10)
    parser.add_argument('--fail-fast-attn-violation', action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def choose_importance(name: str):
    if name == 'group_magnitude':
        if hasattr(tp.importance, 'GroupMagnitudeImportance'):
            return tp.importance.GroupMagnitudeImportance(p=2)
        return tp.importance.MagnitudeImportance(p=2)

    if name == 'magnitude':
        return tp.importance.MagnitudeImportance(p=2)

    if hasattr(tp.importance, 'GroupTaylorImportance'):
        return tp.importance.GroupTaylorImportance()
    if hasattr(tp.importance, 'TaylorImportance'):
        return tp.importance.TaylorImportance()
    raise RuntimeError('Taylor importance is not available in current torch_pruning version.')


def normalize_stage_ids(stage_ids: List[int]) -> List[int]:
    out = []
    for s in stage_ids:
        if 1 <= int(s) <= 4:
            out.append(int(s))
    return sorted(set(out))


def is_in_selected_stage(module_name: str, stage_ids: List[int]) -> bool:
    # SegFormer backbone.layers.{0,1,2,3} corresponds to stages {1,2,3,4}
    for s in stage_ids:
        idx = s - 1
        if module_name.startswith(f'backbone.layers.{idx}'):
            return True
    return False


def build_target_module_names(model: nn.Module, mode: str, prune_stages: List[int]) -> List[str]:
    targets = []

    for name, module in model.named_modules():
        if not is_in_selected_stage(name, prune_stages):
            continue

        if mode == 'mlp_bottleneck':
            # mmseg SegFormer: MixFFN uses Conv2d, and the expansion layer is ffn.layers.0
            if isinstance(module, nn.Conv2d) and (
                name.endswith('ffn.layers.0') or
                name.endswith('ffn.layers.1') or
                name.endswith('ffn.layers.4')
            ):
                targets.append(name)

        elif mode == 'uniform_linear':
            # broader pruning over stage-internal Linear/Conv2d, excluding heads
            if 'decode_head' in name or 'auxiliary_head' in name:
                continue
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                targets.append(name)

    return sorted(set(targets))


def collect_ignored_layers(
    model: nn.Module,
    target_module_names: List[str],
    protect_keywords: List[str],
) -> Tuple[List[nn.Module], List[str]]:
    ignored = []
    ignored_names = []
    target_module_names = set(target_module_names)

    num_classes = None
    if hasattr(model, 'decode_head') and hasattr(model.decode_head, 'num_classes'):
        num_classes = int(model.decode_head.num_classes)

    for name, module in model.named_modules():
        protect = False

        if any(k in name for k in protect_keywords):
            protect = True

        # Only selected modules are allowed to be pruning roots.
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if name not in target_module_names:
                protect = True

        if num_classes is not None:
            if isinstance(module, nn.Conv2d) and module.out_channels == num_classes:
                protect = True
            if isinstance(module, nn.Linear) and module.out_features == num_classes:
                protect = True

        if protect:
            ignored.append(module)
            ignored_names.append(name)

    uniq = {}
    for n, m in zip(ignored_names, ignored):
        uniq[id(m)] = (n, m)
    names = [v[0] for v in uniq.values()]
    layers = [v[1] for v in uniq.values()]
    return layers, names


def validate_attention_dims(model: nn.Module) -> List[str]:
    if not hasattr(model, 'backbone') or not hasattr(model.backbone, 'layers'):
        return []

    violations = []
    for stage_idx, stage in enumerate(model.backbone.layers, start=1):
        try:
            blocks = stage[1]
        except Exception:
            blocks = None
        if blocks is None:
            continue

        for block_idx, blk in enumerate(blocks):
            num_heads = None
            if hasattr(blk, 'attn'):
                num_heads = getattr(blk.attn, 'num_heads', None)
                if num_heads is None and hasattr(blk.attn, 'attn'):
                    num_heads = getattr(blk.attn.attn, 'num_heads', None)

            dim = None
            if hasattr(blk, 'norm1'):
                ns = getattr(blk.norm1, 'normalized_shape', None)
                if isinstance(ns, (list, tuple)) and len(ns) > 0:
                    dim = int(ns[0])

            if num_heads is not None and dim is not None and dim % int(num_heads) != 0:
                violations.append(f'stage{stage_idx}.block{block_idx}: dim={dim}, heads={int(num_heads)}')
    return violations


def build_real_taylor_loader(cfg, work_dir: str):
    runner = build_runner_for_cfg(cfg, work_dir=work_dir)
    return runner.train_dataloader


def accumulate_taylor_gradients(model, train_loader, imp, max_batches: int):
    model.train()
    model.zero_grad(set_to_none=True)

    if hasattr(imp, 'zero_grad'):
        imp.zero_grad()

    for bi, data_batch in enumerate(train_loader):
        if bi >= max_batches:
            break

        processed = model.data_preprocessor(data_batch, training=True)
        losses = model(**processed, mode='loss')
        total_loss = parse_losses(model, losses)

        if hasattr(imp, 'accumulate_grad'):
            # Hessian-style APIs may need per-sample handling. Keep simple here.
            total_loss.backward()
            imp.accumulate_grad(model)
        else:
            total_loss.backward()

    model.eval()

def debug_print_stage_modules(model: nn.Module):
    print("==== Candidate modules in backbone stages ====")
    for name, module in model.named_modules():
        if name.startswith('backbone.layers.') and isinstance(module, (nn.Linear, nn.Conv2d)):
            print(f'{name:<80} {module.__class__.__name__}')
    print("=============================================")

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device.startswith('cuda') and torch.cuda.is_available()) else 'cpu'
    model, cfg = load_segmentor_from_checkpoint(args.config, args.checkpoint, device=device)
    example_inputs = build_example_inputs(tuple(args.shape), device)

    debug_print_stage_modules(model)

    prune_stages = normalize_stage_ids(args.prune_stages)
    target_module_names = build_target_module_names(model, args.mode, prune_stages)
    if len(target_module_names) == 0:
        raise RuntimeError(
            f'No target modules found for mode={args.mode}, prune_stages={prune_stages}. '
            'Check module naming in your SegFormer implementation.'
        )

    imp = choose_importance(args.importance)
    ignored_layers, ignored_names = collect_ignored_layers(
        model=model,
        target_module_names=target_module_names,
        protect_keywords=args.protect_keywords,
    )

    if args.mode == 'mlp_bottleneck':
        root_types = [nn.Conv2d]
    else:
        root_types = [nn.Linear, nn.Conv2d]

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=float(args.pruning_ratio),
        iterative_steps=int(args.iterative_steps),
        max_pruning_ratio=float(args.max_pruning_ratio),
        ignored_layers=ignored_layers,
        round_to=int(args.round_to),
        global_pruning=bool(args.global_pruning),
        isomorphic=bool(args.isomorphic),
        root_module_types=root_types,
        forward_fn=tensor_forward,
    )

    base_macs, base_params = count_ops_and_params(model, example_inputs, forward_fn=tensor_forward)

    history: List[Dict[str, Any]] = []
    history.append({'step': 0, 'macs': base_macs, 'params': base_params})
    attention_warnings: List[Dict[str, Any]] = []

    train_loader = None
    if args.importance == 'taylor':
        train_loader = build_real_taylor_loader(cfg, work_dir=str(output_dir / 'tmp_runner'))

    for step in range(1, args.iterative_steps + 1):
        model_backup = copy.deepcopy(model) if args.fail_fast_attn_violation else None

        if args.importance == 'taylor':
            accumulate_taylor_gradients(model, train_loader, imp, max_batches=int(args.taylor_batches))

        pruner.step()

        violations = validate_attention_dims(model)
        if violations:
            attention_warnings.append({'step': step, 'violations': violations})
            if args.fail_fast_attn_violation and model_backup is not None:
                print(f'[FAIL-FAST] rollback at step={step} because attention dims are invalid.')
                model = model_backup

                rollback_macs, rollback_params = count_ops_and_params(
                    model, example_inputs, forward_fn=tensor_forward
                )

                history.append({
                    'step': step,
                    'rolled_back': True,
                    'violations': violations,
                    'macs': rollback_macs,
                    'params': rollback_params,
                })
                break

        cur_macs, cur_params = count_ops_and_params(model, example_inputs, forward_fn=tensor_forward)
        history.append({
            'step': step,
            'macs': cur_macs,
            'params': cur_params,
            'pruned_params': int(base_params - cur_params),
        })
        print(f'[step {step}] params={cur_params:,}, pruned={base_params-cur_params:,}')

    model.zero_grad(set_to_none=True)

    model_path = output_dir / args.save_model_name
    state_dict_path = output_dir / 'model_pruned_state_dict.pth'
    tp_state_path = output_dir / 'model_pruned_tp_state.pth'
    summary_name = f"summary_{args.mode}_r{str(args.pruning_ratio).replace('.', '')}.json"
    summary_path = output_dir / summary_name


    # For structurally pruned models, full-model saving is the safest for immediate finetuning.
    torch.save(model, model_path)
    torch.save(model.state_dict(), state_dict_path)

    tp_state_saved = False
    try:
        if hasattr(tp, 'state_dict'):
            torch.save(tp.state_dict(model), tp_state_path)
            tp_state_saved = True
    except Exception as e:
        print(f'[WARN] failed to save tp.state_dict: {e}')

    pruned_macs, pruned_params = count_ops_and_params(model, example_inputs, forward_fn=tensor_forward)

    summary = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': device,
        'shape': list(args.shape),
        'mode': args.mode,
        'importance': args.importance,
        'pruning_ratio': float(args.pruning_ratio),
        'iterative_steps': int(args.iterative_steps),
        'max_pruning_ratio': float(args.max_pruning_ratio),
        'round_to': int(args.round_to),
        'global_pruning': bool(args.global_pruning),
        'isomorphic': bool(args.isomorphic),
        'prune_stages': prune_stages,
        'target_module_count': len(target_module_names),
        'target_module_names': target_module_names[:200],
        'ignored_layer_count': len(ignored_layers),
        'ignored_layer_names_preview': ignored_names[:200],
        'attention_warnings': attention_warnings,
        'base_macs': base_macs,
        'base_params': base_params,
        'pruned_macs': pruned_macs,
        'pruned_params': pruned_params,
        'history': history,
        'save_model': str(model_path),
        'save_state_dict': str(state_dict_path),
        'save_tp_state': str(tp_state_path),
        'tp_state_saved': tp_state_saved,
        'notes': {
            'recommendation': 'Use finetune.py with real RailSem19 data immediately after pruning.',
            'taylor_note': 'Taylor uses real segmentation batches here, not dummy random targets.',
        },
    }
    save_json(summary, summary_path)

    print('----------------------------------------')
    print('Summary:')
    if base_macs is not None and pruned_macs is not None:
        print(f'Base MACs: {base_macs/1e9:.2f} G, Pruned MACs: {pruned_macs/1e9:.2f} G')
    print(f'Base Params: {base_params/1e6:.2f} M, Pruned Params: {pruned_params/1e6:.2f} M')
    print(f'Saved model: {model_path}')
    print(f'Saved summary: {summary_path}')


if __name__ == '__main__':
    main()
