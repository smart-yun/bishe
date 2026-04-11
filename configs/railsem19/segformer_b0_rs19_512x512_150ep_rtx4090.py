# -*- coding: utf-8 -*-

# 50-epoch config based on 80k server profile.
# Goal: better RTX4090 throughput + per-epoch metric/checkpoint recording.

_base_ = ['./segformer_b0_rs19_512x512_80000it_server.py']

# RS19 split stats: train.txt has 6800 samples.
# With batch_size=4 => 1700 iters/epoch.
train_size = 6800
train_batch_size = 4
iters_per_epoch = train_size // train_batch_size  # 1700
max_epochs = 150
max_iters = iters_per_epoch * max_epochs  # 85000

# Iter-based training, but validate once per epoch-equivalent.
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=iters_per_epoch)

# Record key training state once per epoch, validate once per epoch,
# and save one checkpoint per epoch (+save_best by mIoU).
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=iters_per_epoch, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=iters_per_epoch, save_best='mIoU'),
)

# RTX4090-friendly data pipeline throughput.
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
)

# Mixed precision for Ada Lovelace GPUs (e.g., RTX4090).
# Since train batch is increased from 2 -> 4, apply linear LR scaling (x2).
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(lr=1.2e-4),
)

# Rebuild scheduler to match the new max_iters.
warmup_iters = 500
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False, begin=warmup_iters, end=max_iters),
]

work_dir = 'runs/rs19/segformer_b0_512x512_50ep_rtx4090'

# Keep weights-only continuation behavior from base config.
resume = False

# Train from scratch: explicitly override inherited load_from from base config.
load_from = None

# Train from scratch: disable pretrained backbone initialization from mmseg base model.
model = dict(
    backbone=dict(init_cfg=None),
)
