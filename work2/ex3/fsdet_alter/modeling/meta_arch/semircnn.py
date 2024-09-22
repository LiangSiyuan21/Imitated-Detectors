# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import _test_optional_float, nn
from torch.nn.modules import pooling
from torchvision.ops.boxes import nms
from fsdet_alter.layers import shape_spec
from fsdet_alter.structures import Boxes
from fsdet_alter.structures import ImageList
from fsdet_alter.utils.logger import log_first_n
import cv2
import random

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from ..instance_search import build_instance_search
from ..roi_heads import build_search_heads
from ..proposal_generator.rpn_outputs import find_top_rpn_proposals
from .build import META_ARCH_REGISTRY

from fsdet_alter.utils.events import get_event_storage
from fsdet_alter.utils.visualizer import Visualizer
from fsdet_alter.data.detection_utils import convert_image_to_rgb, read_image, crop
from fsdet_alter.modeling.utils import concat_all_gathered
from fsdet_alter.modeling.roi_heads.one_shot_detector import fast_rcnn_inference
from ..roi_heads.one_shot_detector import (
    BinaryFastRCNNOutputLayers,
    BinaryFastRCNNOutputs,
    BinaryFastRCNNContrastOutputs,
    BinaryFastRCNNMoCoOutputs,
    BinaryContrastWithPrototypeOutputs,
    BinaryContrastOutputsWithStorage,
    SEARCH_HEADS_OUTPUT_REGISTRY,
)
from fsdet_alter.modeling import instance_search
from fsdet_alter.modeling.hap2s_loss import HAP2STripletLoss


__all__ = ["SemiGeneralizedRCNNv1", "OSGeneralizedRCNNFPN", "SemiGeneralizedRCNNv2", "SemiHAP2SGeneralizedRCNNv1", "SemiHAP2SGeneralizedRCNNv2", "OSHAP2SGeneralizedRCNNFPN", "OSGeneralizedRCNN50", "OSGeneralizedRCNN101"]

cls_names_split1 = ['bird', 'bus', 'cow', 'motorbike', 'sofa']

def load_query(reference_query, choice, choice_list=None, choice_style=None):
    # one-shot
    # Random choice query catgory image
    if choice_list is not None and isinstance(choice_list[0][0], str):
        all_data = reference_query[choice]
        cat_imgs = []
        for i in range(len(all_data)):
            cat_img_path = all_data[i]['image_path']
            cat_imgs.append(cat_img_path)
        # mask_all = [False for _ in range(len(reference_query[choice]))]
        reference_imgs_index = []
        for img_path in choice_list[choice]:
            for j in range(len(cat_imgs)):
                if img_path == cat_imgs[j]:
                    reference_imgs_index.append(j)
        reference_patches_list = []
        # patch_choice_style_1:in_order
        if choice_style == 'every':
            for index in range(len(reference_imgs_index)):
                image = read_image(all_data[index]["image_path"], format="BGR")
                image = crop(image, all_data[index]['bbox'], 128)
                reference_patch = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
                reference_patches_list.append(reference_patch)
        # patch_choice_style_2:random
        else:
            index = random.choice(reference_imgs_index)
            image = read_image(all_data[index]["image_path"], format="BGR")
            image = crop(image, all_data[index]['bbox'], 128)
            reference_patch = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
            reference_patches_list.append(reference_patch)
        
        return reference_patches_list
    else:
        if len(reference_query[choice]) == 0:
            return None
        else:
            all_data = reference_query[choice]
        if choice_list is not None and isinstance(choice_list[0][0], int):
            new_data = []
            for index in choice_list[choice]:
                new_data.append(all_data[index])
            all_data = new_data
        data = random.choice(all_data)
            
        # Get image
        image = read_image(data["image_path"], format="BGR")
        # crop bbox
        image = crop(image, data['bbox'], 128)

        reference_patch = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        return reference_patch

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

@META_ARCH_REGISTRY.register()
class SemiGeneralizedRCNNv1(nn.Module):
    """
    SemiGeneralized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt on #
        self.input_format  = cfg.INPUT.FORMAT
        self.vis_period    = cfg.INPUT.VIS_PERIOD
        self.moco          = cfg.MODEL.MOCO.ENABLED
        # fmt off #

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())  # specify roi_heads name in yaml
        self.instance_search = build_instance_search(cfg, self.backbone.output_shape())
        self.search_heads = build_search_heads(cfg, self.backbone.output_shape()) 
        if self.moco:
            if self.roi_heads.__class__ == 'MoCoROIHeadsV1':
                self.roi_heads._moco_encoder_init(cfg)

            elif self.roi_heads.__class__ == 'MoCoROIHeadsV3':
                self.backbone_k = build_backbone(cfg)
                self.proposal_generator_k = build_proposal_generator(cfg, self.backbone_k.output_shape())
                self.roi_heads_k = build_roi_heads(cfg, self.backbone_k.output_shape())

                self.roi_heads._moco_encoder_init(cfg,
                    self.backbone_k, self.proposal_generator_k, self.roi_heads_k)
        else:
            assert 'MoCo' not in cfg.MODEL.ROI_HEADS.NAME

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.triplet_loss = torch.nn.MarginRankingLoss(margin = -0.3)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.BACKBONE.FREEZE_P5:
            for connection in [self.backbone.fpn_lateral5, self.backbone.fpn_output5]:
                for p in connection.parameters():
                    p.requires_grad = False
            print('frozen P5 in FPN')


        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC2:
                for p in self.roi_heads.box_head.fc2.parameters():
                    p.requires_grad = True
                print('unfreeze fc2 in roi head')

            # we do not ever need to use this in our works.
            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC1:
                for p in self.roi_heads.box_head.fc1.parameters():
                    p.requires_grad = True
                print('unfreeze fc1 in roi head')
        print('-------- Using Roi Head: {}---------\n'.format(cfg.MODEL.ROI_HEADS.NAME))

    def forward(self, batched_inputs, reference_query=None, show_time=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            if reference_query is not None:
                return self.inference(batched_inputs, reference_query=reference_query, show_time=show_time)
            else:
                return self.inference(batched_inputs)

        # backbone FPN
        images = self.preprocess_image(batched_inputs)
        # print(images.tensor)
        features = self.backbone(images.tensor)  # List of L, FPN features
        batched_size = len(batched_inputs)

        # RPN
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]  # List of N
        else:
            gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # self.vis_period = 1
        # tensorboard visualize, visualize top-20 RPN proposals with largest objectness
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        if self.moco and self.roi_heads.__class__ == 'MoCoROIHeadsV2':
            self.roi_heads.gt_instances = gt_instances
        # RoI
        # ROI inputs are post_nms_top_k proposals.
        # detector_losses includes Contrast Loss, 和业务层的 cls loss, and reg loss
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        print(detector_losses)
        # print('gt labels:')
        # print(gt_instances[0].gt_classes)

        # Instance Search
        if "instances_for_can" in batched_inputs[0]:
            can_instances = [x["instances_for_can"].to(self.device) for x in batched_inputs]  # List of N
        else:
            can_instances = None
    

        reference_images = self.preprocess_image(batched_inputs, reference=True)

        reference_features = self.backbone(reference_images.tensor)

        _, act_cans, act_refs, c_weights = self.instance_search(features, reference_features)

        roi_data = self.search_heads.label_and_sample_proposals(proposals, can_instances)


        features_list = act_cans[:-1]
        pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
        pooled_feat = self.search_heads.box_head(pooled_feat)
        
        features_ref_list = act_refs[:-1]

        box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
        boxes_np = box_np.repeat(128, axis=0)  
        query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
        query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
        query_feat = self.search_heads.box_head(query_feat)

        pooled_feat = torch.cat((pooled_feat, query_feat), 1)
        pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
        search_net_outputs = BinaryFastRCNNOutputs(
            self.search_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            roi_data,
            self.search_heads.smooth_l1_beta,
        )
        
        search_net_losses = search_net_outputs.losses()
        
        search_net_losses['loss_retrival_bbox'] = search_net_losses.pop('loss_retrival_bbox') * 1
        search_net_losses['loss_verify_cls'] = search_net_losses.pop('loss_verify_cls') * 1

        score_prob = F.softmax(pred_class_logits, 1)[:,0]

        rois_labels = []
        for x in roi_data:
            labels_list = list(x.gt_classes.cpu().numpy())
            reverse_label = []
            for label in labels_list:
                if label == 0:
                    reverse_label.append(1)
                else:
                    reverse_label.append(0)
            rois_labels.append(np.array(reverse_label))
        rois_labels = torch.Tensor(rois_labels)
      
        score_label = rois_labels.float().cuda()
        gt_map = torch.abs(score_label.unsqueeze(1)-score_label.unsqueeze(-1))

        score_prob = score_prob.view(batched_size, -1)
        pr_map = torch.abs(score_prob.unsqueeze(1)-score_prob.unsqueeze(-1))
        target = -((gt_map-1)**2) + gt_map        

        margin_loss = 3 * self.triplet_loss(pr_map, gt_map, target)
        search_net_losses["margin_loss"] = margin_loss.item()

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                pres, _ = search_net_outputs.inference(score_thresh=0.5, nms_thresh=0.7, topk_per_image=20)
                self.visualize_match_net(batched_inputs, pres)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(search_net_losses)
        return losses

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
           
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net(self, batched_inputs, preds):
        storage = get_event_storage()

        for input, prop in zip(batched_inputs, preds):
            query = input["reference_patch"]
            query = convert_image_to_rgb(query.permute(1, 2, 0), self.input_format)
            img = input["image"]
            (c, h, w) = img.shape
            query = cv2.resize(query, (h, h),interpolation=cv2.INTER_LINEAR)
            query_img = Visualizer(query, None)
            query_img = query_img.img

            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.pred_boxes.tensor.cpu().detach().numpy()
            )
            prop_img = v_pred.get_image()


            vis_img = np.concatenate((query_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: reference patch;  Right: candidate bboxes and cls"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
   
    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True, reference_query=None, show_time=None):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs. 

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        if reference_query is not None:
            real_classes = []
            pred_boxes = []
            scores = []
            # get candidate images
            candidate_images = self.preprocess_image(batched_inputs)
            features = self.backbone(candidate_images.tensor)
            
            if self.proposal_generator:
                # top128
                self.proposal_generator.post_nms_topk = {True:1000, False:128}
                proposals, _ = self.proposal_generator(candidate_images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            
            # one-shot
            choice_list = []
            for index in range(len(reference_query)):
                index_list = []
                cat_len = len(reference_query[index])
                index_list.append(int(cat_len/5)*0);
                choice_list.append(index_list)

            for category in range(len(reference_query)):
                reference_image = load_query(reference_query, category, choice_list)
                if reference_image is None:
                    continue
                reference_features = self.backbone(reference_image.unsqueeze(0).cuda())
                
                _, act_cans, act_refs, c_weights = self.instance_search(features, reference_features)

                roi_data = proposals
                features_list = act_cans[:-1]
                pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
                pooled_feat = self.search_heads.box_head(pooled_feat)
        
                features_ref_list = act_refs[:-1]
                box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
                boxes_np = box_np.repeat(128, axis=0)  
                query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
                query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
                query_feat = self.search_heads.box_head(query_feat)

                pooled_feat = torch.cat((pooled_feat, query_feat), 1)
                pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
                search_net_outputs = BinaryFastRCNNOutputs(
                    self.search_heads.box2box_transform,
                    pred_class_logits,
                    pred_proposal_deltas,
                    roi_data,
                    self.search_heads.smooth_l1_beta,
                )
                result, _ = search_net_outputs.inference(score_thresh=0.5, nms_thresh=0.7, topk_per_image=100)
                results = result
                # temp = result[0].pred_classes.size()
                real_cls = np.ones(result[0].pred_classes.size()) * category
                real_cls_per_cat = real_cls.astype(np.int64).tolist()
                scores_per_cat = result[0].scores.cpu().numpy().tolist()
                pred_boxes_per_cat = result[0].pred_boxes.tensor.cpu().numpy().tolist()
                
                real_classes = real_classes + real_cls_per_cat
                scores = scores + scores_per_cat
                pred_boxes = pred_boxes + pred_boxes_per_cat

            if len(real_classes) == 0:
                return []
            real_classes = torch.Tensor(real_classes).cuda()
            scores = torch.Tensor(scores).cuda()
            pred_boxes = torch.Tensor(pred_boxes).cuda()


            results[0].pred_boxes.tensor = pred_boxes
            results[0].scores = scores
            results[0].pred_classes = real_classes


            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, candidate_images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

        else:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            
            if detected_instances is None:
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                results, _ = self.roi_heads(images, features, proposals, None)
            else:
                detected_instances = [x.to(self.device) for x in detected_instances]
                results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
            
            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

    def preprocess_image(self, batched_inputs, reference=None):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        
        if reference:
            images = [x["reference_patch"].to(self.device) for x in batched_inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

@META_ARCH_REGISTRY.register()
class OSGeneralizedRCNNFPN(nn.Module):
    """
    SemiGeneralized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt on #
        self.input_format  = cfg.INPUT.FORMAT
        self.vis_period    = cfg.INPUT.VIS_PERIOD
        self.moco          = cfg.MODEL.MOCO.ENABLED
        # fmt off #

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())  # specify roi_heads name in yaml
        self.instance_search = build_instance_search(cfg, self.backbone.output_shape())
        self.search_heads = build_search_heads(cfg, self.backbone.output_shape()) 
        if self.moco:
            if self.roi_heads.__class__ == 'MoCoROIHeadsV1':
                self.roi_heads._moco_encoder_init(cfg)

            elif self.roi_heads.__class__ == 'MoCoROIHeadsV3':
                self.backbone_k = build_backbone(cfg)
                self.proposal_generator_k = build_proposal_generator(cfg, self.backbone_k.output_shape())
                self.roi_heads_k = build_roi_heads(cfg, self.backbone_k.output_shape())

                self.roi_heads._moco_encoder_init(cfg,
                    self.backbone_k, self.proposal_generator_k, self.roi_heads_k)
        else:
            assert 'MoCo' not in cfg.MODEL.ROI_HEADS.NAME

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.triplet_loss = torch.nn.MarginRankingLoss(margin = -0.3)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.BACKBONE.FREEZE_P5:
            for connection in [self.backbone.fpn_lateral5, self.backbone.fpn_output5]:
                for p in connection.parameters():
                    p.requires_grad = False
            print('frozen P5 in FPN')


        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC2:
                for p in self.roi_heads.box_head.fc2.parameters():
                    p.requires_grad = True
                print('unfreeze fc2 in roi head')

            # we do not ever need to use this in our works.
            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC1:
                for p in self.roi_heads.box_head.fc1.parameters():
                    p.requires_grad = True
                print('unfreeze fc1 in roi head')
        print('-------- Using Roi Head: {}---------\n'.format(cfg.MODEL.ROI_HEADS.NAME))

    # Only use search net
    def forward(self, batched_inputs, reference_query=None, show_time=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            if reference_query is not None:
                return self.inference(batched_inputs, reference_query=reference_query, show_time=show_time)
            else:
                return self.inference(batched_inputs)

        # backbone FPN
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)  # List of L, FPN features
        batched_size = len(batched_inputs)

        
        reference_images = self.preprocess_image(batched_inputs, reference=True)

        reference_features = self.backbone(reference_images.tensor)

        features, act_cans, act_refs, c_weights = self.instance_search(features, reference_features)
        
        # RPN
        if "instances_for_can" in batched_inputs[0]:
            can_instances = [x["instances_for_can"].to(self.device) for x in batched_inputs]  # List of N
        else:
            can_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, can_instances, reference=True)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # self.vis_period = 1
        # tensorboard visualize, visualize top-20 RPN proposals with largest objectness
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        if self.moco and self.roi_heads.__class__ == 'MoCoROIHeadsV2':
            self.roi_heads.gt_instances = can_instances

        # Instance Search
        # if "instances_for_can" in batched_inputs[0]:
        #     can_instances = [x["instances_for_can"].to(self.device) for x in batched_inputs]  # List of N
        # else:
        #     can_instances = None

        roi_data = self.search_heads.label_and_sample_proposals(proposals, can_instances)


        features_list = act_cans[:-1]
        pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
        pooled_feat = self.search_heads.box_head(pooled_feat)
        
        features_ref_list = act_refs[:-1]

        box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
        boxes_np = box_np.repeat(128, axis=0)  
        query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
        query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
        query_feat = self.search_heads.box_head(query_feat)

        pooled_feat = torch.cat((pooled_feat, query_feat), 1)
        pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
        search_net_outputs = BinaryFastRCNNOutputs(
            self.search_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            roi_data,
            self.search_heads.smooth_l1_beta,
        )
        
        search_net_losses = search_net_outputs.losses()

        score_prob = F.softmax(pred_class_logits, 1)[:,0]

        rois_labels = []
        for x in roi_data:
            labels_list = list(x.gt_classes.cpu().numpy())
            reverse_label = []
            for label in labels_list:
                if label == 0:
                    reverse_label.append(1)
                else:
                    reverse_label.append(0)
            rois_labels.append(np.array(reverse_label))
        rois_labels = torch.Tensor(rois_labels)

        # search_net_losses = {}
        # print('rois_labels')
        # print(rois_labels)         
        score_label = rois_labels.float().cuda()
        gt_map = torch.abs(score_label.unsqueeze(1)-score_label.unsqueeze(-1))

        score_prob = score_prob.view(batched_size, -1)
        pr_map = torch.abs(score_prob.unsqueeze(1)-score_prob.unsqueeze(-1))
        target = -((gt_map-1)**2) + gt_map        

        margin_loss = 3 * self.triplet_loss(pr_map, gt_map, target)
        search_net_losses["margin_loss"] = margin_loss.item()

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                pres, _ = search_net_outputs.inference(score_thresh=0.5, nms_thresh=0.7, topk_per_image=20)
                self.visualize_match_net(batched_inputs, pres)

        losses = {}
        losses.update(proposal_losses)
        losses.update(search_net_losses)
        return losses
    
    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
           
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net(self, batched_inputs, preds):
        storage = get_event_storage()

        for input, prop in zip(batched_inputs, preds):
            query = input["reference_patch"]
            query = convert_image_to_rgb(query.permute(1, 2, 0), self.input_format)
            img = input["image"]
            (c, h, w) = img.shape
            query = cv2.resize(query, (h, h),interpolation=cv2.INTER_LINEAR)
            query_img = Visualizer(query, None)
            query_img = query_img.img

            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.pred_boxes.tensor.cpu().detach().numpy()
            )
            prop_img = v_pred.get_image()


            vis_img = np.concatenate((query_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: reference patch;  Right: candidate bboxes and cls"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    # only search net
    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True, reference_query=None, show_time=None):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs. 

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        if reference_query is not None:
            real_classes = []
            pred_boxes = []
            scores = []
            # get candidate images
            candidate_images = self.preprocess_image(batched_inputs)
            features_init = self.backbone(candidate_images.tensor)
            
            choice_list = []
            for index in range(len(reference_query)):
                index_list = []
                cat_len = len(reference_query[index])
                index_list.append(int(cat_len/5)*0);
                choice_list.append(index_list)

            for category in range(len(reference_query)):
                reference_image = load_query(reference_query, category, choice_list)
                if reference_image is None:
                    continue
                reference_features = self.backbone(reference_image.unsqueeze(0).cuda())
                
                features, act_cans, act_refs, c_weights = self.instance_search(features_init, reference_features)

                if self.proposal_generator:
                    # top128
                    self.proposal_generator.post_nms_topk = {True:1000, False:128}
                    proposals, _ = self.proposal_generator(candidate_images, features, None, reference=True)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                roi_data = proposals
                features_list = act_cans[:-1]
                pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
                pooled_feat = self.search_heads.box_head(pooled_feat)
        
                features_ref_list = act_refs[:-1]
                box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
                boxes_np = box_np.repeat(128, axis=0)  
                query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
                query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
                query_feat = self.search_heads.box_head(query_feat)

                pooled_feat = torch.cat((pooled_feat, query_feat), 1)
                pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
                search_net_outputs = BinaryFastRCNNOutputs(
                    self.search_heads.box2box_transform,
                    pred_class_logits,
                    pred_proposal_deltas,
                    roi_data,
                    self.search_heads.smooth_l1_beta,
                )
                result, _ = search_net_outputs.inference(score_thresh=0.99, nms_thresh=0.7, topk_per_image=100)
                results = result
                # temp = result[0].pred_classes.size()
                real_cls = np.ones(result[0].pred_classes.size()) * category
                real_cls_per_cat = real_cls.astype(np.int64).tolist()
                scores_per_cat = result[0].scores.cpu().numpy().tolist()
                pred_boxes_per_cat = result[0].pred_boxes.tensor.cpu().numpy().tolist()
                
                real_classes = real_classes + real_cls_per_cat
                scores = scores + scores_per_cat
                pred_boxes = pred_boxes + pred_boxes_per_cat

            if len(real_classes) == 0:
                return []
            real_classes = torch.Tensor(real_classes).cuda()
            scores = torch.Tensor(scores).cuda()
            pred_boxes = torch.Tensor(pred_boxes).cuda()


            results[0].pred_boxes.tensor = pred_boxes
            results[0].scores = scores
            results[0].pred_classes = real_classes


            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, candidate_images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

    def preprocess_image(self, batched_inputs, reference=None):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        
        if reference:
            images = [x["reference_patch"].to(self.device) for x in batched_inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

@META_ARCH_REGISTRY.register()
class SemiGeneralizedRCNNv2(nn.Module):
    """
    SemiGeneralized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt on #
        self.input_format  = cfg.INPUT.FORMAT
        self.vis_period    = cfg.INPUT.VIS_PERIOD
        self.moco          = cfg.MODEL.MOCO.ENABLED
        # fmt off #

        # query retrieval setting
        self.shot_num = cfg.TEST.SHOT_NUM
        self.patch_choice = cfg.TEST.PATCH_CHOICE
        self.screening_strategy = cfg.TEST.SCREENING_STRATEGY
        self.xml_img_path = cfg.TEST.SAVE_PATH

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())  # specify roi_heads name in yaml
        self.instance_search = build_instance_search(cfg, self.backbone.output_shape())
        self.search_heads = build_search_heads(cfg, self.backbone.output_shape()) 
        if self.moco:
            if self.roi_heads.__class__ == 'MoCoROIHeadsV1':
                self.roi_heads._moco_encoder_init(cfg)

            elif self.roi_heads.__class__ == 'MoCoROIHeadsV3':
                self.backbone_k = build_backbone(cfg)
                self.proposal_generator_k = build_proposal_generator(cfg, self.backbone_k.output_shape())
                self.roi_heads_k = build_roi_heads(cfg, self.backbone_k.output_shape())

                self.roi_heads._moco_encoder_init(cfg,
                    self.backbone_k, self.proposal_generator_k, self.roi_heads_k)
        else:
            assert 'MoCo' not in cfg.MODEL.ROI_HEADS.NAME

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.triplet_loss = torch.nn.MarginRankingLoss(margin = -0.3)
        self.to(self.device)

        # s1 = 0
        # s2 = 0
        # for model in [self.backbone.fpn_lateral2, self.backbone.fpn_lateral3, self.backbone.fpn_lateral4, self.backbone.fpn_lateral5]:
        #     s1 += sum(p.numel() for p in model.parameters())
        # for model in [self.backbone.fpn_output2, self.backbone.fpn_output3, self.backbone.fpn_output4, self.backbone.fpn_output5]:
        #     s2 += sum(p.numel() for p in model.parameters())
        # print('FPN',s1, s2)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.BACKBONE.FREEZE_P5:
            for connection in [self.backbone.fpn_lateral5, self.backbone.fpn_output5]:
                for p in connection.parameters():
                    p.requires_grad = False
            print('frozen P5 in FPN')


        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC2:
                for p in self.roi_heads.box_head.fc2.parameters():
                    p.requires_grad = True
                print('unfreeze fc2 in roi head')

            # we do not ever need to use this in our works.
            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC1:
                for p in self.roi_heads.box_head.fc1.parameters():
                    p.requires_grad = True
                print('unfreeze fc1 in roi head')
        print('-------- Using Roi Head: {}---------\n'.format(cfg.MODEL.ROI_HEADS.NAME))
    
    def load_queries(self, reference_query, choice, choice_list=None):
        # one-shot
        # Random choice query catgory image
        if choice_list is not None and isinstance(choice_list[0][0], str):
            all_data = reference_query[choice]
            cat_imgs = []
            for i in range(len(all_data)):
                cat_img_path = all_data[i]['image_path']
                cat_imgs.append(cat_img_path)
            # mask_all = [False for _ in range(len(reference_query[choice]))]
            reference_imgs_index = []
            for img_path in choice_list[choice]:
                for j in range(len(cat_imgs)):
                    if img_path == cat_imgs[j]:
                        reference_imgs_index.append(j)
            reference_patches_list = []
            if self.patch_choice == 'random':
                for index in range(len(reference_imgs_index)):
                    image = read_image(all_data[index]["image_path"], format="BGR")
                    image = crop(image, all_data[index]['bbox'], 128)
                    reference_patch = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
                    reference_patches_list.append(reference_patch)
            if self.patch_choice == 'every':
            # patch_choice_style_2:random
                index = random.choice(reference_imgs_index)
                image = read_image(all_data[index]["image_path"], format="BGR")
                image = crop(image, all_data[index]['bbox'], 128)
                reference_patch = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
                reference_patches_list.append(reference_patch)
            
            return reference_patches_list
        else:
            if len(reference_query[choice]) == 0:
                return None
            else:
                all_data = reference_query[choice]
            if choice_list is not None and isinstance(choice_list[0][0], int):
                new_data = []
                for index in choice_list[choice]:
                    new_data.append(all_data[index])
                all_data = new_data
            data = random.choice(all_data)
                
            # Get image
            image = read_image(data["image_path"], format="BGR")
            # crop bbox
            image = crop(image, data['bbox'], 128)

            reference_patch = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

            return reference_patch

    def forward(self, batched_inputs, reference_query=None, show_time=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            if reference_query is not None:
                return self.inference(batched_inputs, reference_query=reference_query, show_time=show_time)
            else:
                return self.inference(batched_inputs)

        # backbone FPN
        images = self.preprocess_image(batched_inputs)
        # print(images.tensor)
        features = self.backbone(images.tensor)  # List of L, FPN features
        batched_size = len(batched_inputs)

        # RPN
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]  # List of N
        else:
            gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # self.vis_period = 1
        # tensorboard visualize, visualize top-20 RPN proposals with largest objectness
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        if self.moco and self.roi_heads.__class__ == 'MoCoROIHeadsV2':
            self.roi_heads.gt_instances = gt_instances
        # RoI
        # ROI inputs are post_nms_top_k proposals.
        # detector_losses includes Contrast Loss, 和业务层的 cls loss, and reg loss
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        print(detector_losses)
        # print('gt labels:')
        # print(gt_instances[0].gt_classes)

        # Instance Search
        if "instances_for_can" in batched_inputs[0]:
            can_instances = [x["instances_for_can"].to(self.device) for x in batched_inputs]  # List of N
        else:
            can_instances = None
    

        reference_images = self.preprocess_image(batched_inputs, reference=True)

        reference_features = self.backbone(reference_images.tensor)

        _, act_cans, act_refs, c_weights = self.instance_search(features, reference_features)

        roi_data = self.search_heads.label_and_sample_proposals(proposals, can_instances)


        features_list = act_cans[:-1]
        pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
        pooled_feat = self.search_heads.box_head(pooled_feat)
        
        features_ref_list = act_refs[:-1]

        box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
        boxes_np = box_np.repeat(128, axis=0)  
        query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
        query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
        query_feat = self.search_heads.box_head(query_feat)

        pooled_feat = torch.cat((pooled_feat, query_feat), 1)
        pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
        search_net_outputs = BinaryFastRCNNOutputs(
            self.search_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            roi_data,
            self.search_heads.smooth_l1_beta,
        )
        
        search_net_losses = search_net_outputs.losses()
        
        search_net_losses['loss_retrival_bbox'] = search_net_losses.pop('loss_retrival_bbox') * 1
        search_net_losses['loss_verify_cls'] = search_net_losses.pop('loss_verify_cls') * 1

        score_prob = F.softmax(pred_class_logits, 1)[:,0]

        rois_labels = []
        for x in roi_data:
            labels_list = list(x.gt_classes.cpu().numpy())
            reverse_label = []
            for label in labels_list:
                if label == 0:
                    reverse_label.append(1)
                else:
                    reverse_label.append(0)
            rois_labels.append(np.array(reverse_label))
        rois_labels = torch.Tensor(rois_labels)
      
        score_label = rois_labels.float().cuda()
        gt_map = torch.abs(score_label.unsqueeze(1)-score_label.unsqueeze(-1))

        score_prob = score_prob.view(batched_size, -1)
        pr_map = torch.abs(score_prob.unsqueeze(1)-score_prob.unsqueeze(-1))
        target = -((gt_map-1)**2) + gt_map        

        margin_loss = 3 * self.triplet_loss(pr_map, gt_map, target)
        search_net_losses["margin_loss"] = margin_loss.item()

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                pres, _ = search_net_outputs.inference(score_thresh=0.5, nms_thresh=0.7, topk_per_image=20)
                self.visualize_match_net(batched_inputs, pres)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(search_net_losses)
        return losses

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
           
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net(self, batched_inputs, preds):
        storage = get_event_storage()

        for input, prop in zip(batched_inputs, preds):
            query = input["reference_patch"]
            query = convert_image_to_rgb(query.permute(1, 2, 0), self.input_format)
            img = input["image"]
            (c, h, w) = img.shape
            query = cv2.resize(query, (h, h),interpolation=cv2.INTER_LINEAR)
            query_img = Visualizer(query, None)
            query_img = query_img.img

            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.pred_boxes.tensor.cpu().detach().numpy()
            )
            prop_img = v_pred.get_image()


            vis_img = np.concatenate((query_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: reference patch;  Right: candidate bboxes and cls"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
   
    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True, reference_query=None, show_time=None):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs. 

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        if reference_query is not None:
            real_classes = []
            pred_boxes = []
            scores = []
            # get candidate images
            candidate_images = self.preprocess_image(batched_inputs)
            features = self.backbone(candidate_images.tensor)
            
            if self.proposal_generator:
                # top128
                self.proposal_generator.post_nms_topk = {True:1000, False:128}
                proposals, _ = self.proposal_generator(candidate_images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            
            # one-shot
            choice_list = []
            for index in range(len(reference_query)):
                index_list = []
                cat_len = len(reference_query[index])
                index_list.append(int(cat_len/5)*0);
                choice_list.append(index_list)
            
            # n-shot
            # seed = str(1);
            # choice_list = []
            # for index in range(len(reference_query)):
            #     cat_list = []
            #     cat_queries = reference_query[index]
            #     cat_name = cls_names_split1[index]
            #     imgs_path = [x['image_path'] for x in cat_queries]
            #     list_path = 'datasets/vocsplit/seed'+seed+'/box_'+str(self.shot_num)+'shot_'+cat_name+'_train.txt'
            #     with open(list_path, 'r') as f:
            #         choice_names = f.read().splitlines()

            #     for i, img_path in enumerate(imgs_path):
            #         if img_path in choice_names:
            #             cat_list.append(i)
            #     choice_list.append(cat_list)

            for category in range(len(reference_query)):
                reference_image = load_query(reference_query, category, choice_list)
                if reference_image is None:
                    continue
                reference_features = self.backbone(reference_image.unsqueeze(0).cuda())
                
                _, act_cans, act_refs, c_weights = self.instance_search(features, reference_features)

                roi_data = proposals
                features_list = act_cans[:-1]
                pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
                pooled_feat = self.search_heads.box_head(pooled_feat)
        
                features_ref_list = act_refs[:-1]
                box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
                boxes_np = box_np.repeat(128, axis=0)  
                query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
                query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
                query_feat = self.search_heads.box_head(query_feat)

                pooled_feat = torch.cat((pooled_feat, query_feat), 1)
                pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
                search_net_outputs = BinaryFastRCNNOutputs(
                    self.search_heads.box2box_transform,
                    pred_class_logits,
                    pred_proposal_deltas,
                    roi_data,
                    self.search_heads.smooth_l1_beta,
                )
                result, _ = search_net_outputs.inference(score_thresh=0.5, nms_thresh=0.7, topk_per_image=100)
                results = result
                # temp = result[0].pred_classes.size()
                real_cls = np.ones(result[0].pred_classes.size()) * category
                real_cls_per_cat = real_cls.astype(np.int64).tolist()
                scores_per_cat = result[0].scores.cpu().numpy().tolist()
                pred_boxes_per_cat = result[0].pred_boxes.tensor.cpu().numpy().tolist()
                
                real_classes = real_classes + real_cls_per_cat
                scores = scores + scores_per_cat
                pred_boxes = pred_boxes + pred_boxes_per_cat

            if len(real_classes) == 0:
                return []
            real_classes = torch.Tensor(real_classes).cuda()
            scores = torch.Tensor(scores).cuda()
            pred_boxes = torch.Tensor(pred_boxes).cuda()


            results[0].pred_boxes.tensor = pred_boxes
            results[0].scores = scores
            results[0].pred_classes = real_classes


            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, candidate_images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

        else:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            
            if detected_instances is None:
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                results, _ = self.roi_heads(images, features, proposals, None)
            else:
                detected_instances = [x.to(self.device) for x in detected_instances]
                results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
            
            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

    def preprocess_image(self, batched_inputs, reference=None):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        
        if reference:
            images = [x["reference_patch"].to(self.device) for x in batched_inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

@META_ARCH_REGISTRY.register()
class SemiHAP2SGeneralizedRCNNv1(nn.Module):
    """
    SemiGeneralized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt on #
        self.input_format  = cfg.INPUT.FORMAT
        self.vis_period    = cfg.INPUT.VIS_PERIOD
        self.moco          = cfg.MODEL.MOCO.ENABLED
        # fmt off #

        # query retrieval setting
        self.shot_num = cfg.TEST.SHOT_NUM
        self.patch_choice = cfg.TEST.PATCH_CHOICE
        self.screening_strategy = cfg.TEST.SCREENING_STRATEGY
        # self.retrieval_txt_path = cfg.TEST.SAVE_PATH
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())  # specify roi_heads name in yaml
        self.instance_search = build_instance_search(cfg, self.backbone.output_shape())
        self.search_heads = build_search_heads(cfg, self.backbone.output_shape()) 
        if self.moco:
            if self.roi_heads.__class__ == 'MoCoROIHeadsV1':
                self.roi_heads._moco_encoder_init(cfg)

            elif self.roi_heads.__class__ == 'MoCoROIHeadsV3':
                self.backbone_k = build_backbone(cfg)
                self.proposal_generator_k = build_proposal_generator(cfg, self.backbone_k.output_shape())
                self.roi_heads_k = build_roi_heads(cfg, self.backbone_k.output_shape())

                self.roi_heads._moco_encoder_init(cfg,
                    self.backbone_k, self.proposal_generator_k, self.roi_heads_k)
        else:
            assert 'MoCo' not in cfg.MODEL.ROI_HEADS.NAME

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.triplet_loss = torch.nn.MarginRankingLoss(margin = -0.3)
        self.hap2s_loss = HAP2STripletLoss(margin=-0.3)
        self.instance_search_flag = not(cfg.MODEL.INSTANCE_SEARCH.FREEZE)
        self.seed = 2;
        if cfg.TEST.SAVE_PATH is not None:
            self.retrieval_txt_path = cfg.TEST.SAVE_PATH + '/shot' + str(self.shot_num) + '/seed' + str(self.seed)
        else:
            self.retrieval_txt_path = None
        self.to(self.device)

        # s1 = 0
        # s2 = 0
        # for model in [self.backbone.fpn_lateral2, self.backbone.fpn_lateral3, self.backbone.fpn_lateral4, self.backbone.fpn_lateral5]:
        #     s1 += sum(p.numel() for p in model.parameters())
        # for model in [self.backbone.fpn_output2, self.backbone.fpn_output3, self.backbone.fpn_output4, self.backbone.fpn_output5]:
        #     s2 += sum(p.numel() for p in model.parameters())
        # print('FPN',s1, s2)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.BACKBONE.FREEZE_P5:
            for connection in [self.backbone.fpn_lateral5, self.backbone.fpn_output5]:
                for p in connection.parameters():
                    p.requires_grad = False
            print('frozen P5 in FPN')

        if cfg.MODEL.INSTANCE_SEARCH.FREEZE:
            for p in self.instance_search.parameters():
                p.requires_grad = False
            print('froze instance search parameters')
            for p in self.search_heads.parameters():
                p.requires_grad = False
            print('froze search head parameters')        
               
        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC2:
                for p in self.roi_heads.box_head.fc2.parameters():
                    p.requires_grad = True
                print('unfreeze fc2 in roi head')

            # we do not ever need to use this in our works.
            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC1:
                for p in self.roi_heads.box_head.fc1.parameters():
                    p.requires_grad = True
                print('unfreeze fc1 in roi head')
        print('-------- Using Roi Head: {}---------\n'.format(cfg.MODEL.ROI_HEADS.NAME))

    def forward(self, batched_inputs, reference_query=None, show_time=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            if reference_query is not None:
                return self.inference(batched_inputs, reference_query=reference_query, show_time=show_time)
            else:
                return self.inference(batched_inputs)

        # backbone FPN
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)  # List of L, FPN features
        batched_size = len(batched_inputs)
        
        # RPN
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]  # List of N
        else:
            gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # self.vis_period = 1
        # tensorboard visualize, visualize top-20 RPN proposals with largest objectness
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        if self.moco and self.roi_heads.__class__ == 'MoCoROIHeadsV2':
            self.roi_heads.gt_instances = gt_instances
        # RoI
        # ROI inputs are post_nms_top_k proposals.
        # detector_losses includes Contrast Loss, 和业务层的 cls loss, and reg loss
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)


        if self.instance_search_flag:
            # Instance Search
            if "instances_for_can" in batched_inputs[0]:
                can_instances = [x["instances_for_can"].to(self.device) for x in batched_inputs]  # List of N
            else:
                can_instances = None
        

            reference_images = self.preprocess_image(batched_inputs, reference=True)

            reference_features = self.backbone(reference_images.tensor)

            _, act_cans, act_refs, c_weights = self.instance_search(features, reference_features)

            roi_data = self.search_heads.label_and_sample_proposals(proposals, can_instances)


            features_list = act_cans[:-1]
            pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
            pooled_feat = self.search_heads.box_head(pooled_feat)
            
            features_ref_list = act_refs[:-1]

            box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
            boxes_np = box_np.repeat(128, axis=0)  
            query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
            query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
            query_feat = self.search_heads.box_head(query_feat)

            pooled_feat = torch.cat((pooled_feat, query_feat), 1)

            # get positive and negative images
            features_list = act_cans[:-1]
            anchor_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes[0] for x in roi_data])
            anchor_feat = self.search_heads.box_head(anchor_feat)
            
            # get positive and negative images
            positive_sets = self.preprocess_image(batched_inputs, positive=True)
            
            feat_anchor_postives = []
            for set_id in range(len(positive_sets)):
                positive_features = self.backbone(positive_sets[set_id].tensor)
                _, _, act_positives, c_weights = self.instance_search(features, positive_features)
            
                features_pos_list = act_positives[:-1]

                box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
                boxes_np = box_np.repeat(1, axis=0)  
                positive_box = Boxes(torch.Tensor(boxes_np).cuda()) 

                positive_feat = self.search_heads.box_pooler(features_pos_list, [positive_box for x in range(len(roi_data))])
                positive_feat = self.search_heads.box_head(positive_feat)
                
                # temp = positive_feat.cpu().detach().numpy()
                pooled_anchor_pos = torch.cat((anchor_feat, positive_feat), 1)
                feat_anchor_postives.append(pooled_anchor_pos)

            negative_sets = self.preprocess_image(batched_inputs, negative=True)
            
            feat_anchor_negatives = []
            for set_id in range(len(negative_sets)):
                negative_features = self.backbone(negative_sets[set_id].tensor)
                _, _, act_negatives, c_weights = self.instance_search(features, negative_features)
            
                features_neg_list = act_negatives[:-1]

                box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
                boxes_np = box_np.repeat(1, axis=0)  
                negative_box = Boxes(torch.Tensor(boxes_np).cuda()) 

                negative_feat = self.search_heads.box_pooler(features_neg_list, [negative_box for x in range(len(roi_data))])
                negative_feat = self.search_heads.box_head(negative_feat)
                
                pooled_anchor_neg = torch.cat((anchor_feat, negative_feat), 1)
                feat_anchor_negatives.append(pooled_anchor_neg)
            
            loss_collect = 0
            for num in range(batched_size):
                anchor_labels = list()
                for set_id in range(len(feat_anchor_postives)):
                    if set_id == 0:
                        feats = feat_anchor_postives[set_id][num].unsqueeze(0)
                    else:
                        feats = torch.cat((feats, feat_anchor_postives[set_id][num].unsqueeze(0)), 0)
                    
                    anchor_labels.append(0)
                
                for set_id in range(len(feat_anchor_negatives)):
                    feats = torch.cat((feats, feat_anchor_negatives[set_id][num].unsqueeze(0)), 0)
                    anchor_labels.append(1)
                # anchor_feats[num] = feats
                anchor_labels = torch.Tensor(anchor_labels).cuda()
                triplet_loss = self.hap2s_loss(feats, anchor_labels)
                loss_collect += triplet_loss.item() * 0.5
            del feat_anchor_negatives, feat_anchor_postives
            loss_collect = loss_collect / batched_size

            
            pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
            search_net_outputs = BinaryFastRCNNOutputs(
                self.search_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                roi_data,
                self.search_heads.smooth_l1_beta,
            )
            
            search_net_losses = search_net_outputs.losses()
            
            search_net_losses['loss_retrival_bbox'] = search_net_losses.pop('loss_retrival_bbox') * 1
            search_net_losses['loss_verify_cls'] = search_net_losses.pop('loss_verify_cls') * 1

            score_prob = F.softmax(pred_class_logits, 1)[:,0]

            rois_labels = []
            for x in roi_data:
                labels_list = list(x.gt_classes.cpu().numpy())
                reverse_label = []
                for label in labels_list:
                    if label == 0:
                        reverse_label.append(1)
                    else:
                        reverse_label.append(0)
                rois_labels.append(np.array(reverse_label))
            rois_labels = torch.Tensor(rois_labels)
        
            score_label = rois_labels.float().cuda()
            gt_map = torch.abs(score_label.unsqueeze(1)-score_label.unsqueeze(-1))

            score_prob = score_prob.view(batched_size, -1)
            pr_map = torch.abs(score_prob.unsqueeze(1)-score_prob.unsqueeze(-1))
            target = -((gt_map-1)**2) + gt_map        

            margin_loss = 3 * self.triplet_loss(pr_map, gt_map, target)
            search_net_losses["margin_loss"] = margin_loss.item()
            search_net_losses["triplet_loss"] = loss_collect

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    pres, _ = search_net_outputs.inference(score_thresh=0.5, nms_thresh=0.7, topk_per_image=20)
                    self.visualize_match_net(batched_inputs, pres)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(search_net_losses)
            return losses
        else:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
           
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net(self, batched_inputs, preds):
        storage = get_event_storage()

        for input, prop in zip(batched_inputs, preds):
            query = input["positive_patches"][0]
            query = convert_image_to_rgb(query.permute(1, 2, 0), self.input_format)
            img = input["image"]
            (c, h, w) = img.shape
            query = cv2.resize(query, (h, h),interpolation=cv2.INTER_LINEAR)
            query_img = Visualizer(query, None)
            query_img = query_img.img

            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.pred_boxes.tensor.cpu().detach().numpy()
            )
            prop_img = v_pred.get_image()


            vis_img = np.concatenate((query_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: reference patch;  Right: candidate bboxes and cls"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
   
    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True, reference_query=None, show_time=None):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs. 

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        if reference_query is not None:
            real_classes_list = []
            pred_boxes_list = []
            scores_list = []
            # get candidate images
            candidate_images = self.preprocess_image(batched_inputs)
            features = self.backbone(candidate_images.tensor)
            
            if self.proposal_generator:
                # top128
                self.proposal_generator.post_nms_topk = {True:1000, False:128}
                proposals, _ = self.proposal_generator(candidate_images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            
            # one-shot
            # choice_list = []
            # for index in range(len(reference_query)):
            #     index_list = []
            #     cat_len = len(reference_query[index])
            #     index_list.append(int(cat_len/5)*0);
            #     choice_list.append(index_list)

            # read_shot from vocsplit seed
            shot = str(self.shot_num)
            seed = str(self.seed)
            choice_list = []
            vocsplit_path_seed = 'datasets/vocsplit/seed'+seed
            for index in range(len(reference_query)):
                cat_name = cls_names_split1[index]
                cat_shot_path = vocsplit_path_seed+'/box_'+shot+'shot_'+cat_name+'_train.txt'
                with open(cat_shot_path, 'r') as f:
                    cat_name_list = f.read().splitlines()
                choice_list.append(cat_name_list)


            for category in range(len(reference_query)):
                if category != 4:
                    continue;
                if choice_list is not None and isinstance(choice_list[0][0], str):
                    #N-shot
                    reference_images_list = load_query(reference_query, category, choice_list, choice_style=self.patch_choice)
                    #inference patch in or
                    for ref_index, reference_image in enumerate(reference_images_list):
                        reference_features = self.backbone(reference_image.unsqueeze(0).cuda())
                    
                        rpn_features, act_cans, act_refs, c_weights = self.instance_search(features, reference_features)

                        if self.proposal_generator:
                            # top128
                            self.proposal_generator.post_nms_topk = {True:1000, False:128}
                            proposals, _ = self.proposal_generator(candidate_images, rpn_features, reference=True)
                        else:
                            assert "proposals" in batched_inputs[0]
                            proposals = [x["proposals"].to(self.device) for x in batched_inputs]


                        roi_data = proposals
                        features_list = act_cans[:-1]
                        pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
                        pooled_feat = self.search_heads.box_head(pooled_feat)
                
                        features_ref_list = act_refs[:-1]
                        box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
                        boxes_np = box_np.repeat(128, axis=0)  
                        query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
                        query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
                        query_feat = self.search_heads.box_head(query_feat)

                        pooled_feat = torch.cat((pooled_feat, query_feat), 1)
                        pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
                        search_net_outputs = BinaryFastRCNNOutputs(
                            self.search_heads.box2box_transform,
                            pred_class_logits,
                            pred_proposal_deltas,
                            roi_data,
                            self.search_heads.smooth_l1_beta,
                        )
                        
                        # if self.screening_strategy == "reverse":
                            # result, _ = search_net_outputs.inference(score_thresh=0.90, nms_thresh=0.7, topk_per_image=100)
                        # elif self.screening_strategy == "filter_fp":
                            # result, _ = search_net_outputs.inference(score_thresh=0.50, nms_thresh=0.7, topk_per_image=10)
                        boxes = search_net_outputs.predict_boxes()
                        scores = search_net_outputs.predict_probs()
                        image_shapes = search_net_outputs.image_shapes
                        if ref_index == 0:
                            n_boxes = boxes[0]
                            n_scores = scores[0]
                        else:
                            n_boxes = torch.cat((n_boxes, boxes[0]), axis=0)
                            n_scores = torch.cat((n_scores, scores[0]), axis=0)
                    if self.screening_strategy == "reverse":
                        result, _ = fast_rcnn_inference(n_boxes.unsqueeze(0), n_scores.unsqueeze(0), image_shapes,score_thresh=0.90, nms_thresh=0.7, topk_per_image=100)
                    if self.screening_strategy == "filter_fp":
                        result, _ = fast_rcnn_inference(n_boxes.unsqueeze(0), n_scores.unsqueeze(0), image_shapes,score_thresh=0.50, nms_thresh=0.7, topk_per_image=10)
                        if len(result[0].scores.cpu().numpy().tolist()) != 0:
                            if result[0].scores.cpu().numpy().tolist()[-1] < 0.8:
                                result, _ = fast_rcnn_inference(n_boxes.unsqueeze(0), n_scores.unsqueeze(0), image_shapes,score_thresh=1, nms_thresh=0.7, topk_per_image=10)
                    results = result
                    # temp = result[0].pred_classes.size()
                    real_cls = np.ones(result[0].pred_classes.size()) * category
                    real_cls_per_cat = real_cls.astype(np.int64).tolist()
                    scores_per_cat = result[0].scores.cpu().numpy().tolist()
                    pred_boxes_per_cat = result[0].pred_boxes.tensor.cpu().numpy().tolist()
                    
                    real_classes = real_classes_list + real_cls_per_cat
                    scores = scores_list + scores_per_cat
                    pred_boxes = pred_boxes_list + pred_boxes_per_cat
                else:
                    #one-shot
                    reference_image = load_query(reference_query, category, choice_list)
                    
                    if reference_image is None:
                        continue
                    
                    reference_features = self.backbone(reference_image.unsqueeze(0).cuda())
                
                    rpn_features, act_cans, act_refs, c_weights = self.instance_search(features, reference_features)

                    if self.proposal_generator:
                        # top128
                        self.proposal_generator.post_nms_topk = {True:1000, False:128}
                        proposals, _ = self.proposal_generator(candidate_images, rpn_features, reference=True)
                    else:
                        assert "proposals" in batched_inputs[0]
                        proposals = [x["proposals"].to(self.device) for x in batched_inputs]


                    roi_data = proposals
                    features_list = act_cans[:-1]
                    pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
                    pooled_feat = self.search_heads.box_head(pooled_feat)
            
                    features_ref_list = act_refs[:-1]
                    box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
                    boxes_np = box_np.repeat(128, axis=0)  
                    query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
                    query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
                    query_feat = self.search_heads.box_head(query_feat)

                    pooled_feat = torch.cat((pooled_feat, query_feat), 1)
                    pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
                    search_net_outputs = BinaryFastRCNNOutputs(
                        self.search_heads.box2box_transform,
                        pred_class_logits,
                        pred_proposal_deltas,
                        roi_data,
                        self.search_heads.smooth_l1_beta,
                    )
                    result, _ = search_net_outputs.inference(score_thresh=0.90, nms_thresh=0.7, topk_per_image=100)
                    results = result
                    # temp = result[0].pred_classes.size()
                    real_cls = np.ones(result[0].pred_classes.size()) * category
                    real_cls_per_cat = real_cls.astype(np.int64).tolist()
                    scores_per_cat = result[0].scores.cpu().numpy().tolist()
                    pred_boxes_per_cat = result[0].pred_boxes.tensor.cpu().numpy().tolist()
                    
                    real_classes = real_classes_list + real_cls_per_cat
                    scores = scores_list + scores_per_cat
                    pred_boxes = pred_boxes_list + pred_boxes_per_cat
                

            if len(real_classes) == 0:
                return []
            real_classes = torch.Tensor(real_classes).cuda()
            scores = torch.Tensor(scores).cuda()
            pred_boxes = torch.Tensor(pred_boxes).cuda()


            results[0].pred_boxes.tensor = pred_boxes
            results[0].scores = scores
            results[0].pred_classes = real_classes


            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, candidate_images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

        else:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            
            if detected_instances is None:
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                results, _ = self.roi_heads(images, features, proposals, None)
            else:
                detected_instances = [x.to(self.device) for x in detected_instances]
                results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
            
            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results
    
    def preprocess_image(self, batched_inputs, reference=None, positive=None, negative=None):
        """
        Normalize, pad and batch the input images.
        """
        if reference:
            images = [x["positive_patches"][0].to(self.device) for x in batched_inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        elif positive:
            images_set = []
            for index in range(len(batched_inputs[0]["positive_patches"])):
                images = [x["positive_patches"][index].to(self.device) for x in batched_inputs]
                images = [self.normalizer(x) for x in images]
                images = ImageList.from_tensors(images, self.backbone.size_divisibility)
                images_set.append(images)
            return images_set
        elif negative:
            images_set = []
            for index in range(len(batched_inputs[0]["negative_patches"])):
                images = [x["negative_patches"][index].to(self.device) for x in batched_inputs]
                images = [self.normalizer(x) for x in images]
                images = ImageList.from_tensors(images, self.backbone.size_divisibility)
                images_set.append(images)
            return images_set
        else:
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        
        return images

@META_ARCH_REGISTRY.register()
class SemiHAP2SGeneralizedRCNNv2(nn.Module):
    """
    SemiGeneralized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt on #
        self.input_format  = cfg.INPUT.FORMAT
        self.vis_period    = cfg.INPUT.VIS_PERIOD
        self.moco          = cfg.MODEL.MOCO.ENABLED
        # fmt off #

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())  # specify roi_heads name in yaml
        self.instance_search = build_instance_search(cfg, self.backbone.output_shape())
        self.search_heads = build_search_heads(cfg, self.backbone.output_shape()) 
        if self.moco:
            if self.roi_heads.__class__ == 'MoCoROIHeadsV1':
                self.roi_heads._moco_encoder_init(cfg)

            elif self.roi_heads.__class__ == 'MoCoROIHeadsV3':
                self.backbone_k = build_backbone(cfg)
                self.proposal_generator_k = build_proposal_generator(cfg, self.backbone_k.output_shape())
                self.roi_heads_k = build_roi_heads(cfg, self.backbone_k.output_shape())

                self.roi_heads._moco_encoder_init(cfg,
                    self.backbone_k, self.proposal_generator_k, self.roi_heads_k)
        else:
            assert 'MoCo' not in cfg.MODEL.ROI_HEADS.NAME

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.triplet_loss = torch.nn.MarginRankingLoss(margin = -0.3)
        self.hap2s_loss = HAP2STripletLoss(margin=-0.3)

        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.BACKBONE.FREEZE_P5:
            for connection in [self.backbone.fpn_lateral5, self.backbone.fpn_output5]:
                for p in connection.parameters():
                    p.requires_grad = False
            print('frozen P5 in FPN')


        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC2:
                for p in self.roi_heads.box_head.fc2.parameters():
                    p.requires_grad = True
                print('unfreeze fc2 in roi head')

            # we do not ever need to use this in our works.
            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC1:
                for p in self.roi_heads.box_head.fc1.parameters():
                    p.requires_grad = True
                print('unfreeze fc1 in roi head')
        print('-------- Using Roi Head: {}---------\n'.format(cfg.MODEL.ROI_HEADS.NAME))
    
    # share the rpn + co-excition
    def forward(self, batched_inputs, reference_query=None, show_time=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            if reference_query is not None:
                return self.inference(batched_inputs, reference_query=reference_query, show_time=show_time)
            else:
                return self.inference(batched_inputs)

        # backbone FPN
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)  # List of L, FPN features
        batched_size = len(batched_inputs)
        
        # RPN
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]  # List of N
        else:
            gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # self.vis_period = 1
        # tensorboard visualize, visualize top-20 RPN proposals with largest objectness
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        if self.moco and self.roi_heads.__class__ == 'MoCoROIHeadsV2':
            self.roi_heads.gt_instances = gt_instances
        # RoI
        # ROI inputs are post_nms_top_k proposals.
        # detector_losses includes Contrast Loss, 和业务层的 cls loss, and reg loss
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        # Instance Search
        if "instances_for_can" in batched_inputs[0]:
            can_instances = [x["instances_for_can"].to(self.device) for x in batched_inputs]  # List of N
        else:
            can_instances = None
    

        reference_images = self.preprocess_image(batched_inputs, reference=True)

        reference_features = self.backbone(reference_images.tensor)

        rpn_features, act_cans, act_refs, c_weights = self.instance_search(features, reference_features)

        if self.proposal_generator:
            can_proposals, can_proposal_losses = self.proposal_generator(images, rpn_features, can_instances, reference=True)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        can_proposal_losses['can_loss_rpn_cls'] = can_proposal_losses.pop('loss_rpn_cls')
        can_proposal_losses['can_ loss_rpn_loc'] = can_proposal_losses.pop('loss_rpn_loc')

        roi_data = self.search_heads.label_and_sample_proposals(can_proposals, can_instances)


        features_list = act_cans[:-1]
        pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
        pooled_feat = self.search_heads.box_head(pooled_feat)
        
        features_ref_list = act_refs[:-1]
        box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
        boxes_np = box_np.repeat(128, axis=0)  
        query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
        query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
        query_feat = self.search_heads.box_head(query_feat)

        pooled_feat = torch.cat((pooled_feat, query_feat), 1)

        # get positive and negative images
        features_list = act_cans[:-1]
        anchor_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes[0] for x in roi_data])
        anchor_feat = self.search_heads.box_head(anchor_feat)
        
        # get positive and negative images
        positive_sets = self.preprocess_image(batched_inputs, positive=True)
        
        feat_anchor_postives = []
        for set_id in range(len(positive_sets)):
            positive_features = self.backbone(positive_sets[set_id].tensor)
            _, _, act_positives, c_weights = self.instance_search(features, positive_features)
        
            features_pos_list = act_positives[:-1]

            box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
            boxes_np = box_np.repeat(1, axis=0)  
            positive_box = Boxes(torch.Tensor(boxes_np).cuda()) 

            positive_feat = self.search_heads.box_pooler(features_pos_list, [positive_box for x in range(len(roi_data))])
            positive_feat = self.search_heads.box_head(positive_feat)
            
            # temp = positive_feat.cpu().detach().numpy()
            pooled_anchor_pos = torch.cat((anchor_feat, positive_feat), 1)
            feat_anchor_postives.append(pooled_anchor_pos)

        negative_sets = self.preprocess_image(batched_inputs, negative=True)
        
        feat_anchor_negatives = []
        for set_id in range(len(negative_sets)):
            negative_features = self.backbone(negative_sets[set_id].tensor)
            _, _, act_negatives, c_weights = self.instance_search(features, negative_features)
        
            features_neg_list = act_negatives[:-1]

            box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
            boxes_np = box_np.repeat(1, axis=0)  
            negative_box = Boxes(torch.Tensor(boxes_np).cuda()) 

            negative_feat = self.search_heads.box_pooler(features_neg_list, [negative_box for x in range(len(roi_data))])
            negative_feat = self.search_heads.box_head(negative_feat)
            
            pooled_anchor_neg = torch.cat((anchor_feat, negative_feat), 1)
            feat_anchor_negatives.append(pooled_anchor_neg)
        
        # anchor_labels = list()
        # anchor_feats = {}
        loss_collect = 0
        for num in range(batched_size):
            anchor_labels = list()
            for set_id in range(len(feat_anchor_postives)):
                if set_id == 0:
                    feats = feat_anchor_postives[set_id][num].unsqueeze(0)
                else:
                    feats = torch.cat((feats, feat_anchor_postives[set_id][num].unsqueeze(0)), 0)
                
                anchor_labels.append(0)
            
            for set_id in range(len(feat_anchor_negatives)):
                feats = torch.cat((feats, feat_anchor_negatives[set_id][num].unsqueeze(0)), 0)
                anchor_labels.append(1)
            # anchor_feats[num] = feats
            anchor_labels = torch.Tensor(anchor_labels).cuda()
            triplet_loss = self.hap2s_loss(feats, anchor_labels)
            loss_collect += triplet_loss.item() * 0.5
        del feat_anchor_negatives, feat_anchor_postives
        loss_collect = loss_collect / batched_size


        pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
        search_net_outputs = BinaryFastRCNNOutputs(
            self.search_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            roi_data,
            self.search_heads.smooth_l1_beta,
        )
        
        search_net_losses = search_net_outputs.losses()
        
        search_net_losses['loss_retrival_bbox'] = search_net_losses.pop('loss_retrival_bbox').item() * 1
        search_net_losses['loss_verify_cls'] = search_net_losses.pop('loss_verify_cls').item() * 1

        score_prob = F.softmax(pred_class_logits, 1)[:,0]

        rois_labels = []
        for x in roi_data:
            labels_list = list(x.gt_classes.cpu().numpy())
            reverse_label = []
            for label in labels_list:
                if label == 0:
                    reverse_label.append(1)
                else:
                    reverse_label.append(0)
            rois_labels.append(np.array(reverse_label))
        rois_labels = torch.Tensor(rois_labels)
       
        score_label = rois_labels.float().cuda()
        gt_map = torch.abs(score_label.unsqueeze(1)-score_label.unsqueeze(-1))

        score_prob = score_prob.view(batched_size, -1)
        pr_map = torch.abs(score_prob.unsqueeze(1)-score_prob.unsqueeze(-1))
        target = -((gt_map-1)**2) + gt_map        

        margin_loss = 3 * self.triplet_loss(pr_map, gt_map, target)
        search_net_losses["margin_loss"] = margin_loss.item()
        search_net_losses["triplet_loss"] = loss_collect

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                pres, _ = search_net_outputs.inference(score_thresh=0.5, nms_thresh=0.7, topk_per_image=20)
                self.visualize_match_net(batched_inputs, pres)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(can_proposal_losses)
        losses.update(search_net_losses)
        return losses


    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
           
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net(self, batched_inputs, preds):
        storage = get_event_storage()

        for input, prop in zip(batched_inputs, preds):
            query = input["positive_patches"][0]
            query = convert_image_to_rgb(query.permute(1, 2, 0), self.input_format)
            img = input["image"]
            (c, h, w) = img.shape
            query = cv2.resize(query, (h, h),interpolation=cv2.INTER_LINEAR)
            query_img = Visualizer(query, None)
            query_img = query_img.img

            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.pred_boxes.tensor.cpu().detach().numpy()
            )
            prop_img = v_pred.get_image()


            vis_img = np.concatenate((query_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: reference patch;  Right: candidate bboxes and cls"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    # inference rpn+coexcition
    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True, reference_query=None, show_time=None):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs. 

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        if reference_query is not None:
            real_classes = []
            pred_boxes = []
            scores = []
            # get candidate images
            candidate_images = self.preprocess_image(batched_inputs)
            features = self.backbone(candidate_images.tensor)
            
            # one-shot
            choice_list = []
            for index in range(len(reference_query)):
                index_list = []
                cat_len = len(reference_query[index])
                index_list.append(int(cat_len/5)*0);
                choice_list.append(index_list)


            for category in range(len(reference_query)):
                reference_image = load_query(reference_query, category, choice_list)
                if reference_image is None:
                    continue
                reference_features = self.backbone(reference_image.unsqueeze(0).cuda())
                
                rpn_features, act_cans, act_refs, c_weights = self.instance_search(features, reference_features)

                if self.proposal_generator:
                    # top128
                    self.proposal_generator.post_nms_topk = {True:1000, False:128}
                    proposals, _ = self.proposal_generator(candidate_images, rpn_features, reference=True)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]


                roi_data = proposals
                features_list = act_cans[:-1]
                pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
                pooled_feat = self.search_heads.box_head(pooled_feat)
        
                features_ref_list = act_refs[:-1]
                box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
                boxes_np = box_np.repeat(128, axis=0)  
                query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
                query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
                query_feat = self.search_heads.box_head(query_feat)

                pooled_feat = torch.cat((pooled_feat, query_feat), 1)
                pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
                search_net_outputs = BinaryFastRCNNOutputs(
                    self.search_heads.box2box_transform,
                    pred_class_logits,
                    pred_proposal_deltas,
                    roi_data,
                    self.search_heads.smooth_l1_beta,
                )
                result, _ = search_net_outputs.inference(score_thresh=0.90, nms_thresh=0.7, topk_per_image=100)
                results = result
                # temp = result[0].pred_classes.size()
                real_cls = np.ones(result[0].pred_classes.size()) * category
                real_cls_per_cat = real_cls.astype(np.int64).tolist()
                scores_per_cat = result[0].scores.cpu().numpy().tolist()
                pred_boxes_per_cat = result[0].pred_boxes.tensor.cpu().numpy().tolist()
                
                real_classes = real_classes + real_cls_per_cat
                scores = scores + scores_per_cat
                pred_boxes = pred_boxes + pred_boxes_per_cat

            if len(real_classes) == 0:
                return []
            real_classes = torch.Tensor(real_classes).cuda()
            scores = torch.Tensor(scores).cuda()
            pred_boxes = torch.Tensor(pred_boxes).cuda()


            results[0].pred_boxes.tensor = pred_boxes
            results[0].scores = scores
            results[0].pred_classes = real_classes


            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, candidate_images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

        else:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            
            if detected_instances is None:
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                results, _ = self.roi_heads(images, features, proposals, None)
            else:
                detected_instances = [x.to(self.device) for x in detected_instances]
                results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
            
            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

    def preprocess_image(self, batched_inputs, reference=None, positive=None, negative=None):
        """
        Normalize, pad and batch the input images.
        """
        if reference:
            images = [x["positive_patches"][0].to(self.device) for x in batched_inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        elif positive:
            images_set = []
            for index in range(len(batched_inputs[0]["positive_patches"])):
                images = [x["positive_patches"][index].to(self.device) for x in batched_inputs]
                images = [self.normalizer(x) for x in images]
                images = ImageList.from_tensors(images, self.backbone.size_divisibility)
                images_set.append(images)
            return images_set
        elif negative:
            images_set = []
            for index in range(len(batched_inputs[0]["negative_patches"])):
                images = [x["negative_patches"][index].to(self.device) for x in batched_inputs]
                images = [self.normalizer(x) for x in images]
                images = ImageList.from_tensors(images, self.backbone.size_divisibility)
                images_set.append(images)
            return images_set
        else:
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        
        return images


@META_ARCH_REGISTRY.register()
class OSHAP2SGeneralizedRCNNFPN(nn.Module):
    """
    SemiGeneralized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt on #
        self.input_format  = cfg.INPUT.FORMAT
        self.vis_period    = cfg.INPUT.VIS_PERIOD
        self.moco          = cfg.MODEL.MOCO.ENABLED
        # fmt off #

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())  # specify roi_heads name in yaml
        self.instance_search = build_instance_search(cfg, self.backbone.output_shape())
        self.search_heads = build_search_heads(cfg, self.backbone.output_shape()) 
        if self.moco:
            if self.roi_heads.__class__ == 'MoCoROIHeadsV1':
                self.roi_heads._moco_encoder_init(cfg)

            elif self.roi_heads.__class__ == 'MoCoROIHeadsV3':
                self.backbone_k = build_backbone(cfg)
                self.proposal_generator_k = build_proposal_generator(cfg, self.backbone_k.output_shape())
                self.roi_heads_k = build_roi_heads(cfg, self.backbone_k.output_shape())

                self.roi_heads._moco_encoder_init(cfg,
                    self.backbone_k, self.proposal_generator_k, self.roi_heads_k)
        else:
            assert 'MoCo' not in cfg.MODEL.ROI_HEADS.NAME

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.triplet_loss = torch.nn.MarginRankingLoss(margin = -0.3)
        self.hap2s_loss = HAP2STripletLoss(margin=-0.5)

        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.BACKBONE.FREEZE_P5:
            for connection in [self.backbone.fpn_lateral5, self.backbone.fpn_output5]:
                for p in connection.parameters():
                    p.requires_grad = False
            print('frozen P5 in FPN')


        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC2:
                for p in self.roi_heads.box_head.fc2.parameters():
                    p.requires_grad = True
                print('unfreeze fc2 in roi head')

            # we do not ever need to use this in our works.
            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC1:
                for p in self.roi_heads.box_head.fc1.parameters():
                    p.requires_grad = True
                print('unfreeze fc1 in roi head')
        print('-------- Using Roi Head: {}---------\n'.format(cfg.MODEL.ROI_HEADS.NAME))

    # Only use search net
    def forward(self, batched_inputs, reference_query=None, show_time=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            if reference_query is not None:
                return self.inference(batched_inputs, reference_query=reference_query, show_time=show_time)
            else:
                return self.inference(batched_inputs)

        # backbone FPN
        images = self.preprocess_image(batched_inputs)
        features_can = self.backbone(images.tensor)  # List of L, FPN features
        batched_size = len(batched_inputs)

        reference_images = self.preprocess_image(batched_inputs, reference=True)

        reference_features = self.backbone(reference_images.tensor)

        features, act_cans, act_refs, c_weights = self.instance_search(features_can, reference_features)
        
        # RPN
        if "instances_for_can" in batched_inputs[0]:
            can_instances = [x["instances_for_can"].to(self.device) for x in batched_inputs]  # List of N
        else:
            can_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, can_instances, reference=True)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # self.vis_period = 1
        # tensorboard visualize, visualize top-20 RPN proposals with largest objectness
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        if self.moco and self.roi_heads.__class__ == 'MoCoROIHeadsV2':
            self.roi_heads.gt_instances = can_instances

        # Instance Search
        # if "instances_for_can" in batched_inputs[0]:
        #     can_instances = [x["instances_for_can"].to(self.device) for x in batched_inputs]  # List of N
        # else:
        #     can_instances = None

        roi_data = self.search_heads.label_and_sample_proposals(proposals, can_instances)


        features_list = act_cans[:-1]
        pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
        pooled_feat = self.search_heads.box_head(pooled_feat)
        
        features_ref_list = act_refs[:-1]

        box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
        boxes_np = box_np.repeat(128, axis=0)  
        query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
        query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
        query_feat = self.search_heads.box_head(query_feat)

        pooled_feat = torch.cat((pooled_feat, query_feat), 1)
        # get positive and negative images
        features_list = act_cans[:-1]
        anchor_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes[0] for x in roi_data])
        anchor_feat = self.search_heads.box_head(anchor_feat)
        
        # get positive and negative images
        positive_sets = self.preprocess_image(batched_inputs, positive=True)
        
        feat_anchor_postives = []
        for set_id in range(len(positive_sets)):
            positive_features = self.backbone(positive_sets[set_id].tensor)
            _, _, act_positives, c_weights = self.instance_search(features_can, positive_features)
        
            features_pos_list = act_positives[:-1]

            box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
            boxes_np = box_np.repeat(1, axis=0)  
            positive_box = Boxes(torch.Tensor(boxes_np).cuda()) 

            positive_feat = self.search_heads.box_pooler(features_pos_list, [positive_box for x in range(len(roi_data))])
            positive_feat = self.search_heads.box_head(positive_feat)
            
            # temp = positive_feat.cpu().detach().numpy()
            pooled_anchor_pos = torch.cat((anchor_feat, positive_feat), 1)
            feat_anchor_postives.append(pooled_anchor_pos)

        negative_sets = self.preprocess_image(batched_inputs, negative=True)
        
        feat_anchor_negatives = []
        for set_id in range(len(negative_sets)):
            negative_features = self.backbone(negative_sets[set_id].tensor)
            _, _, act_negatives, c_weights = self.instance_search(features_can, negative_features)
        
            features_neg_list = act_negatives[:-1]

            box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
            boxes_np = box_np.repeat(1, axis=0)  
            negative_box = Boxes(torch.Tensor(boxes_np).cuda()) 

            negative_feat = self.search_heads.box_pooler(features_neg_list, [negative_box for x in range(len(roi_data))])
            negative_feat = self.search_heads.box_head(negative_feat)
            
            pooled_anchor_neg = torch.cat((anchor_feat, negative_feat), 1)
            feat_anchor_negatives.append(pooled_anchor_neg)
        
        loss_collect = 0
        for num in range(batched_size):
            anchor_labels = list()
            for set_id in range(len(feat_anchor_postives)):
                if set_id == 0:
                    feats = feat_anchor_postives[set_id][num].unsqueeze(0)
                else:
                    feats = torch.cat((feats, feat_anchor_postives[set_id][num].unsqueeze(0)), 0)
                
                anchor_labels.append(0)
            
            for set_id in range(len(feat_anchor_negatives)):
                feats = torch.cat((feats, feat_anchor_negatives[set_id][num].unsqueeze(0)), 0)
                anchor_labels.append(1)
            # anchor_feats[num] = feats
            anchor_labels = torch.Tensor(anchor_labels).cuda()
            triplet_loss = self.hap2s_loss(feats, anchor_labels)
            loss_collect += triplet_loss.item() * 0.5
        del feat_anchor_negatives, feat_anchor_postives
        loss_collect = loss_collect / batched_size

        pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
        search_net_outputs = BinaryFastRCNNOutputs(
            self.search_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            roi_data,
            self.search_heads.smooth_l1_beta,
        )
        
        search_net_losses = search_net_outputs.losses()

        score_prob = F.softmax(pred_class_logits, 1)[:,0]

        rois_labels = []
        for x in roi_data:
            labels_list = list(x.gt_classes.cpu().numpy())
            reverse_label = []
            for label in labels_list:
                if label == 0:
                    reverse_label.append(1)
                else:
                    reverse_label.append(0)
            rois_labels.append(np.array(reverse_label))
        rois_labels = torch.Tensor(rois_labels)

        # search_net_losses = {}
        # print('rois_labels')
        # print(rois_labels)         
        score_label = rois_labels.float().cuda()
        gt_map = torch.abs(score_label.unsqueeze(1)-score_label.unsqueeze(-1))

        score_prob = score_prob.view(batched_size, -1)
        pr_map = torch.abs(score_prob.unsqueeze(1)-score_prob.unsqueeze(-1))
        target = -((gt_map-1)**2) + gt_map        

        margin_loss = 3 * self.triplet_loss(pr_map, gt_map, target)
        search_net_losses["margin_loss"] = margin_loss.item()
        search_net_losses["triplet_loss"] = loss_collect

        # if self.vis_period > 0:
        #     storage = get_event_storage()
        #     if storage.iter % self.vis_period == 0:
        #         pres, _ = search_net_outputs.inference(score_thresh=0.5, nms_thresh=0.7, topk_per_image=20)
        #         self.visualize_match_net(batched_inputs, pres)

        losses = {}
        losses.update(proposal_losses)
        losses.update(search_net_losses)
        # losses.update(loss_collect)
        return losses
    
    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
           
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net(self, batched_inputs, preds):
        storage = get_event_storage()

        for input, prop in zip(batched_inputs, preds):
            query = input["reference_patch"]
            query = convert_image_to_rgb(query.permute(1, 2, 0), self.input_format)
            img = input["image"]
            (c, h, w) = img.shape
            query = cv2.resize(query, (h, h),interpolation=cv2.INTER_LINEAR)
            query_img = Visualizer(query, None)
            query_img = query_img.img

            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.pred_boxes.tensor.cpu().detach().numpy()
            )
            prop_img = v_pred.get_image()


            vis_img = np.concatenate((query_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: reference patch;  Right: candidate bboxes and cls"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    # only search net
    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True, reference_query=None, show_time=None):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs. 

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        if reference_query is not None:
            real_classes = []
            pred_boxes = []
            scores = []
            # get candidate images
            candidate_images = self.preprocess_image(batched_inputs)
            features_init = self.backbone(candidate_images.tensor)
            
            choice_list = []
            for index in range(len(reference_query)):
                index_list = []
                cat_len = len(reference_query[index])
                index_list.append(int(cat_len/5)*0);
                choice_list.append(index_list)

            for category in range(len(reference_query)):
                reference_image = load_query(reference_query, category, choice_list)
                if reference_image is None:
                    continue
                reference_features = self.backbone(reference_image.unsqueeze(0).cuda())
                
                features, act_cans, act_refs, c_weights = self.instance_search(features_init, reference_features)

                if self.proposal_generator:
                    # top128
                    self.proposal_generator.post_nms_topk = {True:1000, False:128}
                    proposals, _ = self.proposal_generator(candidate_images, features, None, reference=True)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                roi_data = proposals
                features_list = act_cans[:-1]
                pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
                pooled_feat = self.search_heads.box_head(pooled_feat)
        
                features_ref_list = act_refs[:-1]
                box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
                boxes_np = box_np.repeat(128, axis=0)  
                query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
                query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
                query_feat = self.search_heads.box_head(query_feat)

                pooled_feat = torch.cat((pooled_feat, query_feat), 1)
                pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
                search_net_outputs = BinaryFastRCNNOutputs(
                    self.search_heads.box2box_transform,
                    pred_class_logits,
                    pred_proposal_deltas,
                    roi_data,
                    self.search_heads.smooth_l1_beta,
                )
                result, _ = search_net_outputs.inference(score_thresh=0.99, nms_thresh=0.7, topk_per_image=100)
                results = result
                # temp = result[0].pred_classes.size()
                real_cls = np.ones(result[0].pred_classes.size()) * category
                real_cls_per_cat = real_cls.astype(np.int64).tolist()
                scores_per_cat = result[0].scores.cpu().numpy().tolist()
                pred_boxes_per_cat = result[0].pred_boxes.tensor.cpu().numpy().tolist()
                
                real_classes = real_classes + real_cls_per_cat
                scores = scores + scores_per_cat
                pred_boxes = pred_boxes + pred_boxes_per_cat

            if len(real_classes) == 0:
                return []
            real_classes = torch.Tensor(real_classes).cuda()
            scores = torch.Tensor(scores).cuda()
            pred_boxes = torch.Tensor(pred_boxes).cuda()


            results[0].pred_boxes.tensor = pred_boxes
            results[0].scores = scores
            results[0].pred_classes = real_classes


            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, candidate_images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

    def preprocess_image(self, batched_inputs, reference=None, positive=None, negative=None):
        """
        Normalize, pad and batch the input images.
        """
        if reference:
            images = [x["positive_patches"][0].to(self.device) for x in batched_inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        elif positive:
            images_set = []
            for index in range(len(batched_inputs[0]["positive_patches"])):
                images = [x["positive_patches"][index].to(self.device) for x in batched_inputs]
                images = [self.normalizer(x) for x in images]
                images = ImageList.from_tensors(images, self.backbone.size_divisibility)
                images_set.append(images)
            return images_set
        elif negative:
            images_set = []
            for index in range(len(batched_inputs[0]["negative_patches"])):
                images = [x["negative_patches"][index].to(self.device) for x in batched_inputs]
                images = [self.normalizer(x) for x in images]
                images = ImageList.from_tensors(images, self.backbone.size_divisibility)
                images_set.append(images)
            return images_set
        else:
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        
        return images

@META_ARCH_REGISTRY.register()
class OSGeneralizedRCNN50(nn.Module):
    """
    SemiGeneralized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt on #
        self.input_format  = cfg.INPUT.FORMAT
        self.vis_period    = cfg.INPUT.VIS_PERIOD
        self.moco          = cfg.MODEL.MOCO.ENABLED
        # fmt off #

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())  # specify roi_heads name in yaml
        self.instance_search = build_instance_search(cfg, self.backbone.output_shape())
        self.search_heads = build_search_heads(cfg, self.backbone.output_shape()) 
        if self.moco:
            if self.roi_heads.__class__ == 'MoCoROIHeadsV1':
                self.roi_heads._moco_encoder_init(cfg)

            elif self.roi_heads.__class__ == 'MoCoROIHeadsV3':
                self.backbone_k = build_backbone(cfg)
                self.proposal_generator_k = build_proposal_generator(cfg, self.backbone_k.output_shape())
                self.roi_heads_k = build_roi_heads(cfg, self.backbone_k.output_shape())

                self.roi_heads._moco_encoder_init(cfg,
                    self.backbone_k, self.proposal_generator_k, self.roi_heads_k)
        else:
            assert 'MoCo' not in cfg.MODEL.ROI_HEADS.NAME

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.triplet_loss = torch.nn.MarginRankingLoss(margin = -0.3)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.BACKBONE.FREEZE_P5:
            for connection in [self.backbone.fpn_lateral5, self.backbone.fpn_output5]:
                for p in connection.parameters():
                    p.requires_grad = False
            print('frozen P5 in FPN')


        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC2:
                for p in self.roi_heads.box_head.fc2.parameters():
                    p.requires_grad = True
                print('unfreeze fc2 in roi head')

            # we do not ever need to use this in our works.
            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC1:
                for p in self.roi_heads.box_head.fc1.parameters():
                    p.requires_grad = True
                print('unfreeze fc1 in roi head')
        print('-------- Using Roi Head: {}---------\n'.format(cfg.MODEL.ROI_HEADS.NAME))

    # Only use search net
    def forward(self, batched_inputs, reference_query=None, show_time=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            if reference_query is not None:
                return self.inference(batched_inputs, reference_query=reference_query, show_time=show_time)
            else:
                return self.inference(batched_inputs, reference_query=reference_query, show_time=show_time)

        # backbone FPN
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)  # List of L, FPN features
        batched_size = len(batched_inputs)
        
        if "instances_for_can" in batched_inputs[0]:
            can_instances = [x["instances_for_can"].to(self.device) for x in batched_inputs]  # List of N
        else:
            can_instances = None

        reference_images = self.preprocess_image(batched_inputs, reference=True)

        reference_features = self.backbone(reference_images.tensor)

        features, act_cans, act_refs, c_weights = self.instance_search(features, reference_features)

        # RPN
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, can_instances, reference=True)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}


        
        # tensorboard visualize, visualize top-20 RPN proposals with largest objectness
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        if self.moco and self.roi_heads.__class__ == 'MoCoROIHeadsV2':
            self.roi_heads.gt_instances = gt_instances
        # RoI

        # if it is training phrase, then use ground trubut bboxes for refining
        roi_data = self.search_heads.label_and_sample_proposals(proposals, can_instances)

        # roi pooling
        features_list = act_cans
        pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
        pooled_feat = self.search_heads.box_head(pooled_feat)
        
        features_ref_list = act_refs

        box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
        boxes_np = box_np.repeat(128, axis=0)  
        query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
        query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
        query_feat = self.search_heads.box_head(query_feat)

        pooled_feat = torch.cat((pooled_feat, query_feat), 1)
        pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
        search_net_outputs = BinaryFastRCNNOutputs(
            self.search_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            roi_data,
            self.search_heads.smooth_l1_beta,
        )
        
        search_net_losses = search_net_outputs.losses()

        score_prob = F.softmax(pred_class_logits, 1)[:,0]

        rois_labels = []
        for x in roi_data:
            labels_list = list(x.gt_classes.cpu().numpy())
            reverse_label = []
            for label in labels_list:
                if label == 0:
                    reverse_label.append(1)
                else:
                    reverse_label.append(0)
            rois_labels.append(np.array(reverse_label))
        rois_labels = torch.Tensor(rois_labels)
      
        score_label = rois_labels.float().cuda()
        gt_map = torch.abs(score_label.unsqueeze(1)-score_label.unsqueeze(-1))

        score_prob = score_prob.view(batched_size, -1)
        pr_map = torch.abs(score_prob.unsqueeze(1)-score_prob.unsqueeze(-1))
        target = -((gt_map-1)**2) + gt_map        

        margin_loss = 3 * self.triplet_loss(pr_map, gt_map, target)
        search_net_losses["margin_loss"] = margin_loss

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                pres, _ = search_net_outputs.inference(score_thresh=0.5, nms_thresh=0.7, topk_per_image=20)
                self.visualize_match_net(batched_inputs, pres)

        losses = {}
        # losses.update(proposal_losses)
        losses.update(search_net_losses)
        return losses
   
    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
           
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net(self, batched_inputs, preds):
        storage = get_event_storage()

        for input, prop in zip(batched_inputs, preds):
            query = input["reference_patch"]
            query = convert_image_to_rgb(query.permute(1, 2, 0), self.input_format)
            img = input["image"]
            (c, h, w) = img.shape
            query = cv2.resize(query, (h, h),interpolation=cv2.INTER_LINEAR)
            query_img = Visualizer(query, None)
            query_img = query_img.img

            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.pred_boxes.tensor.cpu().detach().numpy()
            )
            prop_img = v_pred.get_image()


            vis_img = np.concatenate((query_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: reference patch;  Right: candidate bboxes and cls"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    # only search net
    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True, reference_query=None, show_time=None):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs. 

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        if reference_query is not None:
            real_classes = []
            pred_boxes = []
            scores = []
            # get candidate images
            candidate_images = self.preprocess_image(batched_inputs)
            features = self.backbone(candidate_images.tensor)
            

            choice_list = []
            for index in range(len(reference_query)):
                index_list = []
                cat_len = len(reference_query[index])
                index_list.append(int(cat_len/5)*0);
                choice_list.append(index_list)
            
            # one-shot
            for category in range(len(reference_query)):
                reference_image = load_query(reference_query, category, choice_list)
                if reference_image is None:
                    continue
                reference_features = self.backbone(reference_image.unsqueeze(0).cuda())
                
                if len(reference_features) == 0:
                    print('test')
                features, act_cans, act_refs, c_weights = self.instance_search(reference_features, reference_features)

                if self.proposal_generator:
                    # top128
                    self.proposal_generator.post_nms_topk = {True:1000, False:128}
                    proposals, _ = self.proposal_generator(candidate_images, features, reference=True)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                
                roi_data = proposals
                features_list = act_cans
                pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
                pooled_feat = self.search_heads.box_head(pooled_feat)
        
                features_ref_list = act_refs
                box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
                boxes_np = box_np.repeat(128, axis=0)  
                query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
                query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
                query_feat = self.search_heads.box_head(query_feat)

                pooled_feat = torch.cat((pooled_feat, query_feat), 1)
                pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
                search_net_outputs = BinaryFastRCNNOutputs(
                    self.search_heads.box2box_transform,
                    pred_class_logits,
                    pred_proposal_deltas,
                    roi_data,
                    self.search_heads.smooth_l1_beta,
                )
                result, _ = search_net_outputs.inference(score_thresh=0.5, nms_thresh=0.7, topk_per_image=100)
                results = result
                # temp = result[0].pred_classes.size()
                real_cls = np.ones(result[0].pred_classes.size()) * category
                real_cls_per_cat = real_cls.astype(np.int64).tolist()
                scores_per_cat = result[0].scores.cpu().numpy().tolist()
                pred_boxes_per_cat = result[0].pred_boxes.tensor.cpu().numpy().tolist()
                
                real_classes = real_classes + real_cls_per_cat
                scores = scores + scores_per_cat
                pred_boxes = pred_boxes + pred_boxes_per_cat

            if len(real_classes) == 0:
                return []
            real_classes = torch.Tensor(real_classes).cuda()
            scores = torch.Tensor(scores).cuda()
            pred_boxes = torch.Tensor(pred_boxes).cuda()


            results[0].pred_boxes.tensor = pred_boxes
            results[0].scores = scores
            results[0].pred_classes = real_classes


            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, candidate_images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

        else:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            
            if detected_instances is None:
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                results, _ = self.roi_heads(images, features, proposals, None)
            else:
                detected_instances = [x.to(self.device) for x in detected_instances]
                results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
            
            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results
        
        # instance_search_inference
        # reference_list = 

    def preprocess_image(self, batched_inputs, reference=None):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        
        if reference:
            images = [x["reference_patch"].to(self.device) for x in batched_inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


@META_ARCH_REGISTRY.register()
class OSGeneralizedRCNN101(nn.Module):
    """
    SemiGeneralized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt on #
        self.input_format  = cfg.INPUT.FORMAT
        self.vis_period    = cfg.INPUT.VIS_PERIOD
        self.moco          = cfg.MODEL.MOCO.ENABLED
        # fmt off #

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())  # specify roi_heads name in yaml
        self.instance_search = build_instance_search(cfg, self.backbone.output_shape())
        self.search_heads = build_search_heads(cfg, self.backbone.output_shape()) 
        if self.moco:
            if self.roi_heads.__class__ == 'MoCoROIHeadsV1':
                self.roi_heads._moco_encoder_init(cfg)

            elif self.roi_heads.__class__ == 'MoCoROIHeadsV3':
                self.backbone_k = build_backbone(cfg)
                self.proposal_generator_k = build_proposal_generator(cfg, self.backbone_k.output_shape())
                self.roi_heads_k = build_roi_heads(cfg, self.backbone_k.output_shape())

                self.roi_heads._moco_encoder_init(cfg,
                    self.backbone_k, self.proposal_generator_k, self.roi_heads_k)
        else:
            assert 'MoCo' not in cfg.MODEL.ROI_HEADS.NAME

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.triplet_loss = torch.nn.MarginRankingLoss(margin = -0.3)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.BACKBONE.FREEZE_P5:
            for connection in [self.backbone.fpn_lateral5, self.backbone.fpn_output5]:
                for p in connection.parameters():
                    p.requires_grad = False
            print('frozen P5 in FPN')


        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC2:
                for p in self.roi_heads.box_head.fc2.parameters():
                    p.requires_grad = True
                print('unfreeze fc2 in roi head')

            # we do not ever need to use this in our works.
            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC1:
                for p in self.roi_heads.box_head.fc1.parameters():
                    p.requires_grad = True
                print('unfreeze fc1 in roi head')
        print('-------- Using Roi Head: {}---------\n'.format(cfg.MODEL.ROI_HEADS.NAME))

    # Only use search net
    def forward(self, batched_inputs, reference_query=None, show_time=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            if reference_query is not None:
                return self.inference(batched_inputs, reference_query=reference_query, show_time=show_time)
            else:
                return self.inference(batched_inputs, reference_query=reference_query, show_time=show_time)

        # backbone FPN
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)  # List of L, FPN features
        batched_size = len(batched_inputs)
        
        if "instances_for_can" in batched_inputs[0]:
            can_instances = [x["instances_for_can"].to(self.device) for x in batched_inputs]  # List of N
        else:
            can_instances = None

        reference_images = self.preprocess_image(batched_inputs, reference=True)

        reference_features = self.backbone(reference_images.tensor)

        features, act_cans, act_refs, c_weights = self.instance_search(features, reference_features)

        # RPN
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, can_instances, reference=True)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}


        
        # tensorboard visualize, visualize top-20 RPN proposals with largest objectness
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        if self.moco and self.roi_heads.__class__ == 'MoCoROIHeadsV2':
            self.roi_heads.gt_instances = gt_instances
        # RoI

        # if it is training phrase, then use ground trubut bboxes for refining
        roi_data = self.search_heads.label_and_sample_proposals(proposals, can_instances)

        # roi pooling
        features_list = act_cans
        pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
        pooled_feat = self.search_heads.box_head(pooled_feat)
        
        features_ref_list = act_refs

        box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
        boxes_np = box_np.repeat(128, axis=0)  
        query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
        query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
        query_feat = self.search_heads.box_head(query_feat)

        pooled_feat = torch.cat((pooled_feat, query_feat), 1)
        pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
        search_net_outputs = BinaryFastRCNNOutputs(
            self.search_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            roi_data,
            self.search_heads.smooth_l1_beta,
        )
        
        search_net_losses = search_net_outputs.losses()

        score_prob = F.softmax(pred_class_logits, 1)[:,0]

        rois_labels = []
        for x in roi_data:
            labels_list = list(x.gt_classes.cpu().numpy())
            reverse_label = []
            for label in labels_list:
                if label == 0:
                    reverse_label.append(1)
                else:
                    reverse_label.append(0)
            rois_labels.append(np.array(reverse_label))
        rois_labels = torch.Tensor(rois_labels)
      
        score_label = rois_labels.float().cuda()
        gt_map = torch.abs(score_label.unsqueeze(1)-score_label.unsqueeze(-1))

        score_prob = score_prob.view(batched_size, -1)
        pr_map = torch.abs(score_prob.unsqueeze(1)-score_prob.unsqueeze(-1))
        target = -((gt_map-1)**2) + gt_map        

        margin_loss = 3 * self.triplet_loss(pr_map, gt_map, target)
        search_net_losses["margin_loss"] = margin_loss

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                pres, _ = search_net_outputs.inference(score_thresh=0.5, nms_thresh=0.7, topk_per_image=20)
                self.visualize_match_net(batched_inputs, pres)

        losses = {}
        # losses.update(proposal_losses)
        losses.update(search_net_losses)
        return losses
   
    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
           
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net(self, batched_inputs, preds):
        storage = get_event_storage()

        for input, prop in zip(batched_inputs, preds):
            query = input["reference_patch"]
            query = convert_image_to_rgb(query.permute(1, 2, 0), self.input_format)
            img = input["image"]
            (c, h, w) = img.shape
            query = cv2.resize(query, (h, h),interpolation=cv2.INTER_LINEAR)
            query_img = Visualizer(query, None)
            query_img = query_img.img

            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.pred_boxes.tensor.cpu().detach().numpy()
            )
            prop_img = v_pred.get_image()


            vis_img = np.concatenate((query_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: reference patch;  Right: candidate bboxes and cls"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def visualize_match_net_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    # only search net
    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True, reference_query=None, show_time=None):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs. 

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        if reference_query is not None:
            real_classes = []
            pred_boxes = []
            scores = []
            # get candidate images
            candidate_images = self.preprocess_image(batched_inputs)
            features = self.backbone(candidate_images.tensor)
            

            choice_list = []
            for index in range(len(reference_query)):
                index_list = []
                cat_len = len(reference_query[index])
                index_list.append(int(cat_len/5)*0);
                choice_list.append(index_list)
            
            # one-shot
            for category in range(len(reference_query)):
                reference_image = load_query(reference_query, category, choice_list)
                if reference_image is None:
                    continue
                reference_features = self.backbone(reference_image.unsqueeze(0).cuda())
                
                if len(reference_features) == 0:
                    print('test')
                features, act_cans, act_refs, c_weights = self.instance_search(reference_features, reference_features)

                if self.proposal_generator:
                    # top128
                    self.proposal_generator.post_nms_topk = {True:1000, False:128}
                    proposals, _ = self.proposal_generator(candidate_images, features, reference=True)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                
                roi_data = proposals
                features_list = act_cans
                pooled_feat = self.search_heads.box_pooler(features_list, [x.proposal_boxes for x in roi_data])
                pooled_feat = self.search_heads.box_head(pooled_feat)
        
                features_ref_list = act_refs
                box_np = np.array([[0.0, 0.0, 128.0, 128.0]])
                boxes_np = box_np.repeat(128, axis=0)  
                query_proposal_box = Boxes(torch.Tensor(boxes_np).cuda()) 
                query_feat = self.search_heads.box_pooler(features_ref_list, [query_proposal_box for x in range(len(roi_data))])
                query_feat = self.search_heads.box_head(query_feat)

                pooled_feat = torch.cat((pooled_feat, query_feat), 1)
                pred_class_logits, pred_proposal_deltas = self.search_heads.box_predictor(pooled_feat)
                search_net_outputs = BinaryFastRCNNOutputs(
                    self.search_heads.box2box_transform,
                    pred_class_logits,
                    pred_proposal_deltas,
                    roi_data,
                    self.search_heads.smooth_l1_beta,
                )
                result, _ = search_net_outputs.inference(score_thresh=0.5, nms_thresh=0.7, topk_per_image=100)
                results = result
                # temp = result[0].pred_classes.size()
                real_cls = np.ones(result[0].pred_classes.size()) * category
                real_cls_per_cat = real_cls.astype(np.int64).tolist()
                scores_per_cat = result[0].scores.cpu().numpy().tolist()
                pred_boxes_per_cat = result[0].pred_boxes.tensor.cpu().numpy().tolist()
                
                real_classes = real_classes + real_cls_per_cat
                scores = scores + scores_per_cat
                pred_boxes = pred_boxes + pred_boxes_per_cat

            if len(real_classes) == 0:
                return []
            real_classes = torch.Tensor(real_classes).cuda()
            scores = torch.Tensor(scores).cuda()
            pred_boxes = torch.Tensor(pred_boxes).cuda()


            results[0].pred_boxes.tensor = pred_boxes
            results[0].scores = scores
            results[0].pred_classes = real_classes


            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, candidate_images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

        else:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            
            if detected_instances is None:
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                results, _ = self.roi_heads(images, features, proposals, None)
            else:
                detected_instances = [x.to(self.device) for x in detected_instances]
                results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
            
            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results
        
        # instance_search_inference
        # reference_list = 


    def preprocess_image(self, batched_inputs, reference=None):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        
        if reference:
            images = [x["reference_patch"].to(self.device) for x in batched_inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


