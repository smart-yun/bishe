# -*- coding: utf-8 -*-

# Server-oriented 80k training config (inherits baseline 40k config)

_base_ = ['./segformer_b0_rs19_512x512_40000it.py']

max_iters = 80000

# Validate more frequently when searching for best iter
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=1000)

# Keep save_best, but increase checkpoint frequency for safer resume
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2000, save_best='mIoU'),
)

# Rebuild schedulers with the new max_iters
warmup_iters = 500
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False, begin=warmup_iters, end=max_iters),
]

# Avoid tensorboard hard dependency on fresh servers
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer',
)

work_dir = 'runs/rs19/segformer_b0_512x512_80000it_server'

# Continue from 40k weights only (do not resume optimizer/scheduler state)
load_from = 'runs/rs19/segformer_b0_512x512_40000it/iter_40000.pth'
resume = False
