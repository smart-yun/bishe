from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.optim import build_optim_wrapper
from mmengine.runner import Runner

from distill_losses import channel_wise_divergence, kd_kl_div_loss
from utils_mmseg import load_cfg, load_segmentor_from_checkpoint, setup_mmseg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Finetune a structurally pruned SegFormer with teacher-student distillation.'
    )
    parser.add_argument('--config', required=True, help='baseline mmseg config path')
    parser.add_argument('--pruned-model', required=True, help='path to full pruned model object (.pth)')
    parser.add_argument('--teacher-checkpoint', required=True, help='teacher checkpoint path')
    parser.add_argument('--teacher-config', default=None, help='teacher config path; defaults to --config')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--work-dir', required=True)

    parser.add_argument('--finetune-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--val-interval', type=int, default=None)
    parser.add_argument('--save-interval', type=int, default=None)

    parser.add_argument('--distill', choices=['logit', 'cwd', 'logit+cwd'], default='logit+cwd')
    parser.add_argument('--kd-temperature', type=float, default=4.0)
    parser.add_argument('--kd-loss-weight', type=float, default=1.0)
    parser.add_argument('--cwd-tau', type=float, default=1.0)
    parser.add_argument('--cwd-loss-weight', type=float, default=1.0)
    parser.add_argument(
        '--cwd-feature-index',
        type=int,
        default=-1,
        help='which backbone feature to distill; -1 means the last feature map',
    )

    return parser.parse_args()


def resolve_device(device_str: str) -> str:
    if device_str.startswith('cuda') and torch.cuda.is_available():
        return device_str
    return 'cpu'


def load_checkpoint_state_dict(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f'Unexpected checkpoint type from {ckpt_path}: {type(ckpt)}')


def disable_pretrained_init_cfg(node):
    """Recursively disable pretrained/init_cfg loading inside a mmengine Config/ConfigDict."""
    if node is None:
        return

    try:
        if 'pretrained' in node:
            node['pretrained'] = None
    except Exception:
        pass

    try:
        if 'init_cfg' in node:
            node['init_cfg'] = None
    except Exception:
        pass

    try:
        items = list(node.items())
    except Exception:
        return

    for _, v in items:
        if isinstance(v, (dict, list, tuple)):
            disable_pretrained_init_cfg(v)


def strip_student_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith('student.'):
            out[k[len('student.'):]] = v
    return out


class KDPrunedSegmentor(BaseModel):
    """A thin teacher-student wrapper around the pruned student model.

    Notes:
        1. This implementation is intentionally lightweight and drop-in.
        2. It computes segmentation loss from the student as usual.
        3. It adds:
           - logit KD on decode logits
           - CWD on one backbone feature map
        4. For mlp_bottleneck pruning in this repo, feature channel dimensions remain
           stable, so CWD is the best first choice.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        *,
        use_logit_kd: bool = True,
        use_cwd: bool = True,
        kd_temperature: float = 4.0,
        kd_loss_weight: float = 1.0,
        cwd_tau: float = 1.0,
        cwd_loss_weight: float = 1.0,
        cwd_feature_index: int = -1,
    ):
        super().__init__(data_preprocessor=student.data_preprocessor, init_cfg=None)
        self.student = student
        self.teacher = teacher
        self.use_logit_kd = use_logit_kd
        self.use_cwd = use_cwd
        self.kd_temperature = float(kd_temperature)
        self.kd_loss_weight = float(kd_loss_weight)
        self.cwd_tau = float(cwd_tau)
        self.cwd_loss_weight = float(cwd_loss_weight)
        self.cwd_feature_index = int(cwd_feature_index)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def _get_seg_logits(self, model: nn.Module, feats):
        if not hasattr(model, 'decode_head'):
            raise RuntimeError('The wrapped segmentor has no decode_head.')
        return model.decode_head(feats)

    def _select_feat(self, feats):
        if isinstance(feats, (list, tuple)):
            idx = self.cwd_feature_index
            return feats[idx]
        if torch.is_tensor(feats):
            return feats
        raise TypeError(f'Unsupported feature container: {type(feats)}')

    def loss(self, inputs: torch.Tensor, data_samples=None) -> Dict[str, torch.Tensor]:
        losses = self.student.loss(inputs, data_samples)

        # Minimal-intrusion implementation: compute features/logits once more for KD.
        # This is slower than a hook-optimized implementation, but much easier to
        # integrate into the current repo without rewriting mmseg internals.
        student_feats = self.student.extract_feat(inputs)
        with torch.no_grad():
            teacher_feats = self.teacher.extract_feat(inputs)

        if self.use_logit_kd and self.kd_loss_weight > 0:
            student_logits = self._get_seg_logits(self.student, student_feats)
            with torch.no_grad():
                teacher_logits = self._get_seg_logits(self.teacher, teacher_feats)
            losses['loss_kd_logit'] = self.kd_loss_weight * kd_kl_div_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                tau=self.kd_temperature,
            )

        if self.use_cwd and self.cwd_loss_weight > 0:
            student_feat = self._select_feat(student_feats)
            teacher_feat = self._select_feat(teacher_feats)
            losses['loss_kd_cwd'] = self.cwd_loss_weight * channel_wise_divergence(
                student_feat=student_feat,
                teacher_feat=teacher_feat,
                tau=self.cwd_tau,
            )

        return losses

    def predict(self, inputs: torch.Tensor, data_samples=None):
        return self.student.predict(inputs, data_samples)

    def _forward(self, inputs: torch.Tensor, data_samples=None):
        return self.student._forward(inputs, data_samples)


def save_full_models(
    work_dir: Path,
    trained_model: BaseModel,
    pruned_model_template: nn.Module,
):
    work_dir.mkdir(parents=True, exist_ok=True)

    latest_full_model = work_dir / 'latest_full_model.pth'
    student_latest = trained_model.student.cpu() if hasattr(trained_model, 'student') else trained_model.cpu()
    torch.save(student_latest, latest_full_model)
    print(f'Saved latest full model to: {latest_full_model}')

    best_candidates = sorted(work_dir.glob('best_mIoU*.pth'))
    if not best_candidates:
        print('[WARN] No best_mIoU*.pth found. Skip saving best_full_model.pth')
        return

    best_ckpt = best_candidates[-1]
    state_dict = load_checkpoint_state_dict(best_ckpt)
    student_state = strip_student_prefix(state_dict)
    if not student_state:
        student_state = state_dict

    best_model = copy.deepcopy(pruned_model_template).cpu()
    best_model.load_state_dict(student_state, strict=True)

    best_full_model = work_dir / 'best_full_model.pth'
    torch.save(best_model, best_full_model)
    print(f'Saved best full model to: {best_full_model}')
    print(f'Matched best checkpoint: {best_ckpt.name}')


def main():
    args = parse_args()
    setup_mmseg()

    cfg = load_cfg(args.config)
    cfg.work_dir = args.work_dir
    cfg.resume = False
    cfg.load_from = None
    disable_pretrained_init_cfg(cfg.model)

    iters_per_epoch = int(getattr(cfg, 'iters_per_epoch', 0))
    if iters_per_epoch <= 0:
        raise RuntimeError(
            'The config must expose `iters_per_epoch` for epoch-equivalent finetuning. '
            'Please add it to your RailSem19 config.'
        )

    ft_max_iters = int(args.finetune_epochs) * iters_per_epoch
    cfg.train_cfg.max_iters = ft_max_iters
    cfg.train_cfg.val_interval = (
        int(args.val_interval) if args.val_interval is not None else iters_per_epoch
    )

    if 'optimizer' in cfg.optim_wrapper:
        cfg.optim_wrapper.optimizer.lr = float(args.lr)
        cfg.optim_wrapper.optimizer.weight_decay = float(args.weight_decay)

    if 'checkpoint' in cfg.default_hooks:
        cfg.default_hooks.checkpoint.by_epoch = False
        cfg.default_hooks.checkpoint.interval = (
            int(args.save_interval) if args.save_interval is not None else iters_per_epoch
        )
        cfg.default_hooks.checkpoint.save_best = 'mIoU'

    if 'logger' in cfg.default_hooks:
        cfg.default_hooks.logger.interval = max(50, iters_per_epoch // 10)
        cfg.default_hooks.logger.log_metric_by_epoch = False

    runner = Runner.from_cfg(cfg)

    device = resolve_device(args.device)
    student = torch.load(args.pruned_model, map_location='cpu')
    if not isinstance(student, nn.Module):
        raise TypeError(f'--pruned-model must be a full torch.nn.Module, got {type(student)}')
    student = student.to(device)

    teacher_config = args.teacher_config or args.config
    teacher, _ = load_segmentor_from_checkpoint(
        teacher_config,
        args.teacher_checkpoint,
        device=device,
    )
    teacher.eval()

    pruned_model_template = copy.deepcopy(student).cpu()

    use_logit_kd = args.distill in ('logit', 'logit+cwd')
    use_cwd = args.distill in ('cwd', 'logit+cwd')

    kd_model = KDPrunedSegmentor(
        student=student,
        teacher=teacher,
        use_logit_kd=use_logit_kd,
        use_cwd=use_cwd,
        kd_temperature=args.kd_temperature,
        kd_loss_weight=args.kd_loss_weight,
        cwd_tau=args.cwd_tau,
        cwd_loss_weight=args.cwd_loss_weight,
        cwd_feature_index=args.cwd_feature_index,
    ).to(device)

    runner.model = kd_model
    runner.optim_wrapper = build_optim_wrapper(kd_model, cfg.optim_wrapper)

    print('----------------------------------------')
    print('KD Finetune setup:')
    print(f'Pruned model: {args.pruned_model}')
    print(f'Teacher checkpoint: {args.teacher_checkpoint}')
    print(f'Work dir: {cfg.work_dir}')
    print(f'Finetune epochs: {args.finetune_epochs}')
    print(f'Max iters: {cfg.train_cfg.max_iters}')
    print(f'Val interval: {cfg.train_cfg.val_interval}')
    print(f'LR: {cfg.optim_wrapper.optimizer.lr}')
    print(f'Distill mode: {args.distill}')
    print(f'KD T: {args.kd_temperature}, KD weight: {args.kd_loss_weight}')
    print(f'CWD tau: {args.cwd_tau}, CWD weight: {args.cwd_loss_weight}')
    print(f'CWD feature index: {args.cwd_feature_index}')
    print('----------------------------------------')

    runner.train()

    save_full_models(
        work_dir=Path(cfg.work_dir),
        trained_model=runner.model.cpu(),
        pruned_model_template=pruned_model_template,
    )


if __name__ == '__main__':
    main()
