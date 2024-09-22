_base_ = './retinanet_r50_fpn_1x_voc_07.py'
# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)


# modified
model = dict(
    bbox_head=dict(num_classes=20),)
evaluation = dict(interval=24)
optimizer = dict(lr=0.01)