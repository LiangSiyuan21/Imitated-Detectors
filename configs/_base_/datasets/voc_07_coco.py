_base_ = './coco_detection.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


data_root = 'data/VOCdevkit/'
data = dict(
    train=dict(
        ann_file=data_root + 'anno_coco/voc07_trainval.json',
        img_prefix=data_root,
        classes=CLASSES,
        pipeline=train_pipeline,
    ),
    val=dict(
        ann_file=data_root + 'anno_coco/voc07_test.json',
        img_prefix=data_root,
        classes=CLASSES,
        pipeline=test_pipeline,
    ),
    test=dict(
        ann_file=data_root + 'anno_coco/voc07_test.json',
        img_prefix=data_root,
        classes=CLASSES,
        pipeline=test_pipeline,
    ),
)