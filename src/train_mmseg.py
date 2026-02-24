import argparse
from mmengine.config import Config
from mmengine.runner import Runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to mmseg config (.py)')
    parser.add_argument('--work-dir', default=None, help='Override work_dir in config')
    args = parser.parse_args()

    # 关键：注册 mmseg 的所有模块（模型/数据集/评估器等）
    from mmseg.utils import register_all_modules
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # 关键：避免分布式 launcher 相关默认值干扰
    cfg.launcher = 'none'

    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
