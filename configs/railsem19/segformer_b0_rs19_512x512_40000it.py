# -*- coding: utf-8 -*-

# SegFormer-B0 baseline for RailSem19 (500 iters sanity run)

_base_ = [
    'mmseg::segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py',
]

# Import project-local modules (custom dataset)
custom_imports = dict(
    imports=['datasets.rs19_mmseg_dataset'],
    allow_failed_imports=False,
)

# -----------------------
# RailSem19 metainfo (from rs19-config.json)
# label id is assumed to be the index order (0..18)
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
ignore_index = 255  # keep as default; adjust if dataset has no 255

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

# ---------- Dataloader（add sampler） ----------
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        ann_file=train_split,
        metainfo=metainfo,
        
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        ann_file=val_split,
        metainfo=metainfo,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='PackSegInputs'),
        ],
    )
)

test_dataloader = val_dataloader

# -----------------------
# Model head
# -----------------------
ignore_index = 255
model = dict(
    decode_head=dict(
        num_classes=num_classes,
        ignore_index=ignore_index,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
            # ignore_index=ignore_index
        ),
    )
)

# -----------------------
# Train schedule: 500 iters sanity run
# -----------------------
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

max_iters = 40000
warmup_iters = 500  # 500 or 1000

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False, begin=warmup_iters, end=max_iters),
]

# mIoU
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# Logging & visualization
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=50),
)


# ---------- TensorBoard（shotscreen） ----------
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
    name='visualizer'
)

work_dir = 'runs/rs19/segformer_b0_512x512_40000it'
