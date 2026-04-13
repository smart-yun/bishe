# -*- coding: utf-8 -*-

# SegFormer-B1 baseline for RailSem19
# 100 epochs, RTX4090-friendly, 512x512, iter-based with epoch-equivalent validation

_base_ = [
    'mmseg::segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py',
]

# Import project-local modules (custom dataset)
custom_imports = dict(
    imports=['datasets.rs19_mmseg_dataset'],
    allow_failed_imports=False,
)

# -----------------------
# RailSem19 metainfo
# label id is assumed to be index order: 0..18
# -----------------------
classes = [
    'road', 'sidewalk', 'construction', 'tram-track', 'fence', 'pole',
    'traffic-light', 'traffic-sign', 'vegetation', 'terrain', 'sky', 'human',
    'rail-track', 'car', 'truck', 'trackbed', 'on-rails', 'rail-raised', 'rail-embedded'
]

palette = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (192, 0, 128),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (230, 150, 140),
    (0, 0, 142),
    (0, 0, 70),
    (90, 40, 40),
    (0, 80, 100),
    (0, 254, 254),
    (0, 68, 63),
]

num_classes = len(classes)
ignore_index = 255
metainfo = dict(classes=classes, palette=palette)

# -----------------------
# Dataset
# -----------------------
dataset_type = 'RS19JpgListDataset'
data_root = 'data/railsem19'
img_dir = 'jpgs'
ann_dir = 'uint8'

train_split = '../splits_mmseg/train.txt'
val_split   = '../splits_mmseg/val.txt'
test_split  = '../splits_mmseg/test.txt'

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        ann_file=train_split,
        metainfo=metainfo,
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        ann_file=val_split,
        metainfo=metainfo,
        pipeline=test_pipeline,
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        ann_file=test_split,
        metainfo=metainfo,
        pipeline=test_pipeline,
    )
)

# -----------------------
# Training schedule
# RS19 train split: 6800 samples
# batch_size = 4 -> 1700 iters/epoch
# 100 epochs -> 170000 iters
# -----------------------
train_size = 6800
train_batch_size = 4
iters_per_epoch = train_size // train_batch_size   # 1700
max_epochs = 100
max_iters = iters_per_epoch * max_epochs           # 170000

train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=iters_per_epoch)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# -----------------------
# Model
# -----------------------
model = dict(
    decode_head=dict(
        num_classes=num_classes,
        ignore_index=ignore_index,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
)

# -----------------------
# Optimization
# Same style as your B0 RTX4090 config
# batch_size from mmseg base 2 -> here 4, so lr uses x2 logic
# -----------------------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1.2e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

warmup_iters = 500
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False, begin=warmup_iters, end=max_iters),
]

# -----------------------
# Evaluation
# -----------------------
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# -----------------------
# Hooks
# Validate and save once per epoch-equivalent
# -----------------------
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=iters_per_epoch, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=iters_per_epoch, save_best='mIoU'),
)

# Optional: if you want visualization during validation, uncomment below
# default_hooks = dict(
#     logger=dict(type='LoggerHook', interval=iters_per_epoch, log_metric_by_epoch=False),
#     checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=iters_per_epoch, save_best='mIoU'),
#     visualization=dict(type='SegVisualizationHook', draw=True, interval=iters_per_epoch),
# )

# -----------------------
# Runtime
# -----------------------
work_dir = 'runs/rs19/segformer_b1_512x512_100ep_rtx4090'

# resume = False

# # Train from scratch:
# # explicitly disable inherited pretrained loading behavior
# load_from = None