# -*- coding: utf-8 -*-

# 100k extension config with mIoU-based early stopping.
# Base on the validated 80k server config.

_base_ = ['./segformer_b0_rs19_512x512_80000it_server.py']

# Keep custom dataset import + add custom hook import.
custom_imports = dict(
    imports=[
        'datasets.rs19_mmseg_dataset',
        'hooks.early_stop_miou',
    ],
    allow_failed_imports=False,
)

max_iters = 100000

# Validate every 1k iters so patience=5 means ~5k iters plateau window.
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=1000)

# Keep frequent checkpointing and best checkpoint tracking.
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2000, save_best='mIoU'),
)

# Rebuild schedulers for 100k horizon.
warmup_iters = 500
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False, begin=warmup_iters, end=max_iters),
]

# Early stop rule from discussion:
# stop if mIoU improvement < 0.05 for 5 consecutive validations.
custom_hooks = [
    dict(
        type='EarlyStopMIOUHook',
        monitor='mIoU',
        min_delta=0.05,
        patience=5,
        rule='greater',
    )
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer',
)

work_dir = 'runs/rs19/segformer_b0_512x512_100000it_earlystop'

# Continue from 80k best checkpoint weights only.
load_from = 'runs/rs19/segformer_b0_512x512_80000it_server/best_mIoU_iter_79000.pth'
resume = False
