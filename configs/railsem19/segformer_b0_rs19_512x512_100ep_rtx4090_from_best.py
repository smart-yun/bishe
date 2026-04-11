# -*- coding: utf-8 -*-

# 100-epoch continuation config from 50-epoch best checkpoint.
# Train from best weights (weights-only load), not strict resume state.

_base_ = ['./segformer_b0_rs19_512x512_50ep_rtx4090.py']

train_size = 6800
train_batch_size = 4
iters_per_epoch = train_size // train_batch_size  # 1700
max_epochs = 100
max_iters = iters_per_epoch * max_epochs  # 170000

train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=iters_per_epoch)

# Keep per-epoch validation/checkpoint/logger.
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=iters_per_epoch, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=iters_per_epoch, save_best='mIoU'),
)

# Rebuild scheduler for 100-epoch horizon.
warmup_iters = 500
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False, begin=warmup_iters, end=max_iters),
]

work_dir = 'runs/rs19/segformer_b0_512x512_100ep_rtx4090_from_best'

# Weights-only continuation from current best of 50ep run.
load_from = 'runs/rs19/segformer_b0_512x512_50ep_rtx4090/best_mIoU_iter_81600.pth'
resume = False
