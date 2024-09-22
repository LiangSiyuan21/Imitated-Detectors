from ..builder import DETECTORS
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint
import os
import json
import torch
import random

@DETECTORS.register_module()
class PsdDetector(BaseDetector):
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 teacher_pretrained=None,
                 gt_ratio=50,
                 lb_map_path=None,
                 score_thr=0.3,
                 ):
        super(PsdDetector, self).__init__()

        # if lb_map_path != None  and os.path.exists(lb_map_path):
        #     self.lb_map = json.load(open(lb_map_path, 'r'))
        # else:
        #     self.lb_map={}
        # self.gt_ratio=gt_ratio
        self.score_thr = score_thr

        self.teacher = build_detector(teacher_cfg.model,
                                      train_cfg=teacher_cfg.get('train_cfg'),
                                      test_cfg=teacher_cfg.get('test_cfg'))
        self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()

        self.student = build_detector(student_cfg.model,
                                      train_cfg=student_cfg.get('train_cfg'),
                                      test_cfg=student_cfg.get('test_cfg'))
        self.student.init_weights()

    def forward_train(self, imgs, img_metas, **kwargs):
        with torch.no_grad():
            # bboxes = []
            # labels = []

            # self.teacher.eval()
            pred_bboxes = []
            pred_labels = []
            feat = self.teacher.extract_feat(imgs)
            # outs = self.teacher.bbox_head(feat)
            bbox_list = self.teacher.bbox_head.simple_test(
                feat, img_metas, rescale=False)
            for i in range(len(bbox_list)):
                scores = bbox_list[i][0][:, -1]
                # filter bbox with low score
                inds = scores > self.score_thr
                pred_bbox = bbox_list[i][0][inds, :-1].detach().cuda()
                pred_label = bbox_list[i][1][inds].detach().cuda()
                pred_bboxes.append(pred_bbox)
                pred_labels.append(pred_label)

            # gt labels
            # gt_bboxes = kwargs.get('gt_bboxes')
            # gt_labels = kwargs.get('gt_labels')
            #
            # assert self.gt_ratio >= 0
            # for i in range(len(img_metas)):
            #     if img_metas[i]['ori_filename'] not in self.lb_map.keys():
            #         dice = random.randint(0, 99)
            #         if not self.pseudo:
            #             dice = -1
            #         if dice < self.gt_ratio:
            #             self.lb_map[img_metas[i]['ori_filename']] = True
            #             self.gt_count += 1
            #         else:
            #             self.lb_map[img_metas[i]['ori_filename']] = False
            #     get_label_from_gt = self.lb_map[img_metas[i]['ori_filename']]
            #     if get_label_from_gt:
            #         bbox = gt_bboxes[i]
            #         label = gt_labels[i]
            #     else:
            #         bbox = pred_bboxes[i]
            #         label = pred_labels[i]
            #     bboxes.append(bbox)
            #     labels.append(label)

        super(PsdDetector, self).forward_train(imgs, img_metas)

        student_loss = self.student.forward_train(imgs, img_metas, gt_bboxes=pred_bboxes, gt_labels=pred_labels)
        return student_loss

    def base_parameters(self):
        return self.student

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')


    def extract_feat(self, imgs):
        return self.student.extract_feat(imgs)

    async def async_simple_test(self, img, img_metas, **kwargs):
        return self.student.async_simple_test(img, img_metas, **kwargs)

    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)