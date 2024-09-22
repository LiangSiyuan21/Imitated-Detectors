_base_ = [
    '../../_base_/datasets/voc_0712_coco.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True
weight=0.0
pred_weight=0.5
tau=1.0
model=dict()
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = '/srv/hdd/weight/model_stealing/teacher_voc0712_e12.pth',
    distill_cfg = [ 
                    dict(
                         methods=[dict(
                                       name='pred_loss',
                                       weight =pred_weight,
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/retinanet/retinanet_r50_fpn_1x_voc_0712.py'
teacher_cfg = 'configs/retinanet/retinanet_x101_64x4d_fpn_1x_voc_0712.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)
