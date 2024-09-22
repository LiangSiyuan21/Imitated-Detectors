_base_ = './retinanet_r50_fpn_1x_coco_20.py'
model = dict(
    pretrained='open-mmlab://resnext101_64x4d',
    bbox_head=dict(num_classes=20),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'
        # init_cfg=dict(
        #     type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')
        ))

evaluation = dict(interval=24)
optimizer = dict(lr=0.01)