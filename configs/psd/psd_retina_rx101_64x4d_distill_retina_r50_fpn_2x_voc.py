_base_ = [
    '../_base_/datasets/voc_0712_coco.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True

student_cfg = '/home/jwl/code/MS_voc/configs/retinanet/retinanet_r50_fpn_2x_voc_0712.py'
teacher_cfg = '/home/jwl/code/MS_voc/configs/retinanet/retinanet_x101_64x4d_fpn_2x_voc_0712.py'
model=dict(
    type='MSDetector',
    teacher_pretrained = "/home/jwl/code/MS_voc/result/teacher_voc/latest.pth"
)

optimizer = dict(type='SGD', lr=0.01/8, momentum=0.9, weight_decay=0.0001)
evaluation = dict(interval=24)