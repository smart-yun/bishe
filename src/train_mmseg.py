# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to mmseg config (.py)')
    parser.add_argument('--work-dir', default=None, help='Override work_dir in config')
<<<<<<< HEAD
    parser.add_argument('--resume-from', default=None, help='Resume training from checkpoint path (恢复优化器/调度器状态)')
    parser.add_argument('--load-from', default=None, help='Load model weights only (不恢复优化器/调度器状态)')
=======
    parser.add_argument('--resume-from', default=None, help='Resume training from checkpoint path')
    parser.add_argument('--load-from', default=None, help='Load model weights only (no optimizer/scheduler state)')
>>>>>>> c19feb0 (wip: save local changes before pull)
    args = parser.parse_args()

    # 关键：注册 mmseg 的所有模块（模型/数据集/评估器等）
    from mmseg.utils import register_all_modules
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    if args.resume_from is not None and args.load_from is not None:
<<<<<<< HEAD
        raise ValueError('--resume-from and --load-from cannot be used together')
=======
        raise ValueError('Use only one of --resume-from or --load-from')
>>>>>>> c19feb0 (wip: save local changes before pull)

    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Resume checkpoint not found: {ckpt_path}')
        cfg.resume = True
        cfg.load_from = str(ckpt_path)

    if args.load_from is not None:
        ckpt_path = Path(args.load_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Load checkpoint not found: {ckpt_path}')
        cfg.resume = False
        cfg.load_from = str(ckpt_path)

    # 关键：避免分布式 launcher 相关默认值干扰
    cfg.launcher = 'none'

    runner = Runner.from_cfg(cfg)

    # 某些服务器环境下，LocalVisBackend 在写入标量时目录可能尚未创建，
    # 这里提前确保 work_dir/<timestamp>/vis_data 存在，避免 FileNotFoundError。
    try:
        timestamp = getattr(runner, 'timestamp', None)
        if timestamp:
            vis_dir = Path(runner.work_dir) / str(timestamp) / 'vis_data'
            vis_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f'[WARN] Failed to pre-create vis_data directory: {e}')

    runner.train()

if __name__ == '__main__':
    main()
