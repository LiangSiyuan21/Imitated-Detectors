_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco20.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
model = dict(
    bbox_head=dict(num_classes=20),)
evaluation = dict(interval=1)
optimizer = dict(lr=0.01)
