ann_dir = 'uint8'
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'
classes = [
    'road',
    'sidewalk',
    'construction',
    'tram-track',
    'fence',
    'pole',
    'traffic-light',
    'traffic-sign',
    'vegetation',
    'terrain',
    'sky',
    'human',
    'rail-track',
    'car',
    'truck',
    'trackbed',
    'on-rails',
    'rail-raised',
    'rail-embedded',
]
crop_size = (
    512,
    512,
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'datasets.rs19_mmseg_dataset',
    ])
data_preprocessor = dict(
    _scope_='mmseg',
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/railsem19'
dataset_type = 'RS19JpgListDataset'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmseg',
        by_epoch=False,
        interval=1700,
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(
        _scope_='mmseg',
        interval=1700,
        log_metric_by_epoch=False,
        type='LoggerHook'),
    param_scheduler=dict(_scope_='mmseg', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmseg', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmseg', type='IterTimerHook'),
    visualization=dict(
        _scope_='mmseg', draw=True, interval=50, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
ignore_index = 255
img_dir = 'jpgs'
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
iters_per_epoch = 1700
launcher = 'none'
load_from = 'runs/rs19/segformer_b0_512x512_50ep_rtx4090/best_mIoU_iter_81600.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
max_epochs = 100
max_iters = 170000
metainfo = dict(
    classes=[
        'road',
        'sidewalk',
        'construction',
        'tram-track',
        'fence',
        'pole',
        'traffic-light',
        'traffic-sign',
        'vegetation',
        'terrain',
        'sky',
        'human',
        'rail-track',
        'car',
        'truck',
        'trackbed',
        'on-rails',
        'rail-raised',
        'rail-embedded',
    ],
    palette=[
        (
            128,
            64,
            128,
        ),
        (
            244,
            35,
            232,
        ),
        (
            70,
            70,
            70,
        ),
        (
            192,
            0,
            128,
        ),
        (
            190,
            153,
            153,
        ),
        (
            153,
            153,
            153,
        ),
        (
            250,
            170,
            30,
        ),
        (
            220,
            220,
            0,
        ),
        (
            107,
            142,
            35,
        ),
        (
            152,
            251,
            152,
        ),
        (
            70,
            130,
            180,
        ),
        (
            220,
            20,
            60,
        ),
        (
            230,
            150,
            140,
        ),
        (
            0,
            0,
            142,
        ),
        (
            0,
            0,
            70,
        ),
        (
            90,
            40,
            40,
        ),
        (
            0,
            80,
            100,
        ),
        (
            0,
            254,
            254,
        ),
        (
            0,
            68,
            63,
        ),
    ])
model = dict(
    _scope_='mmseg',
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=32,
        in_channels=3,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            2,
            2,
            2,
            2,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        ignore_index=255,
        in_channels=[
            32,
            64,
            160,
            256,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=19,
        type='SegformerHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(_scope_='mmseg', requires_grad=True, type='SyncBN')
num_classes = 19
optim_wrapper = dict(
    _scope_='mmseg',
    clip_grad=dict(max_norm=1.0, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.00012, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='AmpOptimWrapper')
optimizer = dict(
    _scope_='mmseg', lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
palette = [
    (
        128,
        64,
        128,
    ),
    (
        244,
        35,
        232,
    ),
    (
        70,
        70,
        70,
    ),
    (
        192,
        0,
        128,
    ),
    (
        190,
        153,
        153,
    ),
    (
        153,
        153,
        153,
    ),
    (
        250,
        170,
        30,
    ),
    (
        220,
        220,
        0,
    ),
    (
        107,
        142,
        35,
    ),
    (
        152,
        251,
        152,
    ),
    (
        70,
        130,
        180,
    ),
    (
        220,
        20,
        60,
    ),
    (
        230,
        150,
        140,
    ),
    (
        0,
        0,
        142,
    ),
    (
        0,
        0,
        70,
    ),
    (
        90,
        40,
        40,
    ),
    (
        0,
        80,
        100,
    ),
    (
        0,
        254,
        254,
    ),
    (
        0,
        68,
        63,
    ),
]
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=1e-06, type='LinearLR'),
    dict(
        begin=500,
        by_epoch=False,
        end=170000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = dict(_scope_='mmseg', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmseg',
        ann_file='../splits_mmseg/val.txt',
        data_prefix=dict(img_path='jpgs', seg_map_path='uint8'),
        data_root='data/railsem19',
        metainfo=dict(
            classes=[
                'road',
                'sidewalk',
                'construction',
                'tram-track',
                'fence',
                'pole',
                'traffic-light',
                'traffic-sign',
                'vegetation',
                'terrain',
                'sky',
                'human',
                'rail-track',
                'car',
                'truck',
                'trackbed',
                'on-rails',
                'rail-raised',
                'rail-embedded',
            ],
            palette=[
                (
                    128,
                    64,
                    128,
                ),
                (
                    244,
                    35,
                    232,
                ),
                (
                    70,
                    70,
                    70,
                ),
                (
                    192,
                    0,
                    128,
                ),
                (
                    190,
                    153,
                    153,
                ),
                (
                    153,
                    153,
                    153,
                ),
                (
                    250,
                    170,
                    30,
                ),
                (
                    220,
                    220,
                    0,
                ),
                (
                    107,
                    142,
                    35,
                ),
                (
                    152,
                    251,
                    152,
                ),
                (
                    70,
                    130,
                    180,
                ),
                (
                    220,
                    20,
                    60,
                ),
                (
                    230,
                    150,
                    140,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    0,
                    0,
                    70,
                ),
                (
                    90,
                    40,
                    40,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    0,
                    254,
                    254,
                ),
                (
                    0,
                    68,
                    63,
                ),
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='RS19JpgListDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(_scope_='mmseg', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmseg', iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(_scope_='mmseg', type='LoadImageFromFile'),
    dict(_scope_='mmseg', keep_ratio=True, scale=(
        2048,
        512,
    ), type='Resize'),
    dict(_scope_='mmseg', reduce_zero_label=True, type='LoadAnnotations'),
    dict(_scope_='mmseg', type='PackSegInputs'),
]
test_split = '../splits_mmseg/test.txt'
train_batch_size = 4
train_cfg = dict(
    _scope_='mmseg',
    max_iters=170000,
    type='IterBasedTrainLoop',
    val_interval=1700)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        _scope_='mmseg',
        ann_file='../splits_mmseg/train.txt',
        data_prefix=dict(img_path='jpgs', seg_map_path='uint8'),
        data_root='data/railsem19',
        metainfo=dict(
            classes=[
                'road',
                'sidewalk',
                'construction',
                'tram-track',
                'fence',
                'pole',
                'traffic-light',
                'traffic-sign',
                'vegetation',
                'terrain',
                'sky',
                'human',
                'rail-track',
                'car',
                'truck',
                'trackbed',
                'on-rails',
                'rail-raised',
                'rail-embedded',
            ],
            palette=[
                (
                    128,
                    64,
                    128,
                ),
                (
                    244,
                    35,
                    232,
                ),
                (
                    70,
                    70,
                    70,
                ),
                (
                    192,
                    0,
                    128,
                ),
                (
                    190,
                    153,
                    153,
                ),
                (
                    153,
                    153,
                    153,
                ),
                (
                    250,
                    170,
                    30,
                ),
                (
                    220,
                    220,
                    0,
                ),
                (
                    107,
                    142,
                    35,
                ),
                (
                    152,
                    251,
                    152,
                ),
                (
                    70,
                    130,
                    180,
                ),
                (
                    220,
                    20,
                    60,
                ),
                (
                    230,
                    150,
                    140,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    0,
                    0,
                    70,
                ),
                (
                    90,
                    40,
                    40,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    0,
                    254,
                    254,
                ),
                (
                    0,
                    68,
                    63,
                ),
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    512,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='RS19JpgListDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(_scope_='mmseg', shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(_scope_='mmseg', type='LoadImageFromFile'),
    dict(_scope_='mmseg', reduce_zero_label=True, type='LoadAnnotations'),
    dict(
        _scope_='mmseg',
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2048,
            512,
        ),
        type='RandomResize'),
    dict(
        _scope_='mmseg',
        cat_max_ratio=0.75,
        crop_size=(
            512,
            512,
        ),
        type='RandomCrop'),
    dict(_scope_='mmseg', prob=0.5, type='RandomFlip'),
    dict(_scope_='mmseg', type='PhotoMetricDistortion'),
    dict(_scope_='mmseg', type='PackSegInputs'),
]
train_size = 6800
train_split = '../splits_mmseg/train.txt'
tta_model = dict(_scope_='mmseg', type='SegTTAModel')
tta_pipeline = [
    dict(_scope_='mmseg', backend_args=None, type='LoadImageFromFile'),
    dict(
        _scope_='mmseg',
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(_scope_='mmseg', type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmseg',
        ann_file='../splits_mmseg/val.txt',
        data_prefix=dict(img_path='jpgs', seg_map_path='uint8'),
        data_root='data/railsem19',
        metainfo=dict(
            classes=[
                'road',
                'sidewalk',
                'construction',
                'tram-track',
                'fence',
                'pole',
                'traffic-light',
                'traffic-sign',
                'vegetation',
                'terrain',
                'sky',
                'human',
                'rail-track',
                'car',
                'truck',
                'trackbed',
                'on-rails',
                'rail-raised',
                'rail-embedded',
            ],
            palette=[
                (
                    128,
                    64,
                    128,
                ),
                (
                    244,
                    35,
                    232,
                ),
                (
                    70,
                    70,
                    70,
                ),
                (
                    192,
                    0,
                    128,
                ),
                (
                    190,
                    153,
                    153,
                ),
                (
                    153,
                    153,
                    153,
                ),
                (
                    250,
                    170,
                    30,
                ),
                (
                    220,
                    220,
                    0,
                ),
                (
                    107,
                    142,
                    35,
                ),
                (
                    152,
                    251,
                    152,
                ),
                (
                    70,
                    130,
                    180,
                ),
                (
                    220,
                    20,
                    60,
                ),
                (
                    230,
                    150,
                    140,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    0,
                    0,
                    70,
                ),
                (
                    90,
                    40,
                    40,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    0,
                    254,
                    254,
                ),
                (
                    0,
                    68,
                    63,
                ),
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='RS19JpgListDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(_scope_='mmseg', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmseg', iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
val_split = '../splits_mmseg/val.txt'
vis_backends = [
    dict(_scope_='mmseg', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmseg',
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
warmup_iters = 500
work_dir = 'runs/rs19/segformer_b0_512x512_100ep_rtx4090_from_best'
