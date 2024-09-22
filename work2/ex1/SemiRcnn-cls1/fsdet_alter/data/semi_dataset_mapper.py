# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
from albumentations.augmentations.geometric.functional import bbox_affine
import numpy as np
from numpy.random.mtrand import choice
import random
import torch
import cv2

import json
import albumentations as A
# from fsdet_alter.data.build import print_instances_class_histogram
# from .build import print_instances_class_histogram
from fsdet_alter.structures import BoxMode

from . import detection_utils as utils
from . import transforms as T

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["SemiDatasetMapper", "AlbumentationMapper"]



class SemiDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic.

    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True, hap2s=None):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        # fmt: on

        self.is_train = is_train
        self.hap2s = hap2s

        self.reference_query = None
        self.show_time = None
    

    def crop(self, image, purpose, size):
    
        h, w, c = image.shape

        # add_h = int(purpose[4]-purpose[2])/5
        # add_w = int(purpose[3]-purpose[1])/5

        # purpose[2] = int(purpose[2])-add_h if (int(purpose[2])-add_h >0) else 0
        # purpose[4] = int(purpose[4])+add_h if (int(purpose[4])+add_h <h) else h

        # purpose[1] = int(purpose[1])-add_w if (int(purpose[1])-add_w >0) else 0
        # purpose[3] = int(purpose[3])+add_w if (int(purpose[3])+add_w <w) else w
        cut_image = image[int(purpose[1]):int(purpose[3]),int(purpose[0]):int(purpose[2]),:]


        height, width = cut_image.shape[0:2]

        max_hw   = max(height, width)
        cty, ctx = [height // 2, width // 2]

        cropped_image  = np.zeros((max_hw, max_hw, 3), dtype=cut_image.dtype)

        x0, x1 = max(0, ctx - max_hw // 2), min(ctx + max_hw // 2, width)
        y0, y1 = max(0, cty - max_hw // 2), min(cty + max_hw // 2, height)

        left, right = ctx - x0, x1 - ctx
        top, bottom = cty - y0, y1 - cty

        cropped_cty, cropped_ctx = max_hw // 2, max_hw // 2
        y_slice = slice(cropped_cty - top, cropped_cty + bottom)
        x_slice = slice(cropped_ctx - left, cropped_ctx + right)
        cropped_image[y_slice, x_slice, :] = cut_image[y0:y1, x0:x1, :]


        return cv2.resize(cropped_image, (size,size), interpolation=cv2.INTER_LINEAR)


    # random choice query catgory
    def select_reference_patch(self, catgory):
        cand = np.unique(catgory)
        if len(cand)==1:
            choice = cand[0]
        else:
            p = []
            for i in cand:
                p.append(self.show_time[i])
            p = np.array(p)
            p /= p.sum()
            choice  = np.random.choice(cand,1,p=p)[0]
        return choice
    
    def load_query(self, choice, id=0):
        if self.is_train:
            # Random choice query catgory image
            all_data = self.reference_query[choice]
            data     = random.choice(all_data)
        else:
            print('!!!!!!!!!!test load query')
        
        # Get image
        image = utils.read_image(data["image_path"], format=self.img_format)
        # crop bbox
        image = self.crop(image, data['bbox'], 128)

        reference_patch = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        return reference_patch

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day

         # random choice query catgory

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

            catgory = dataset_dict["instances"]._fields['gt_classes'].numpy().copy()
            choice_patch = self.select_reference_patch(catgory)

            instances_for_can = utils.annotations_to_instances(annos, image_shape, True, choice_patch)
            dataset_dict["instances_for_can"] = instances_for_can

        if self.is_train and self.hap2s is None:
            reference_patch = self.load_query(choice_patch, dataset_dict)
            dataset_dict["reference_patch"] = reference_patch
        elif self.hap2s:
            positive_patches = list()
            negative_patches = list()
            catgory_nums = len(self.reference_query)
            if len(np.unique(catgory)) > 1:
                # all_catgory = np.delete()
                all_catgory = catgory[catgory!=choice_patch]
                random_choice = all_catgory[np.random.randint(0, len(all_catgory))]
                nagetive_choice_patch = int(self.select_reference_patch(np.ones(1)*random_choice))
            else:
                all_catgory = np.delete(np.array([x for x in range(catgory_nums)]), choice_patch)
                random_choice = all_catgory[np.random.randint(0, catgory_nums-1)]
                nagetive_choice_patch = int(self.select_reference_patch(np.ones(1)*random_choice))

            for _ in range(4):
                reference_patch = self.load_query(choice_patch, dataset_dict)
                positive_patches.append(reference_patch)

                reference_patch = self.load_query(nagetive_choice_patch, dataset_dict)
                negative_patches.append(reference_patch)
            dataset_dict['positive_patches'] = positive_patches
            dataset_dict['negative_patches'] = negative_patches
            
        return dataset_dict


class AlbumentationMapper:
    debug_count = 5
    def __init__(self, cfg, is_train=True):
        # use the detectron2 crop_gen
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None
        # default detectron2 tfm_gens contains ResizeShortestEdge and RandomFlip(horizontal only by default)
        self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        # load albumnentation json
        logging.getLogger(__name__).info("Albumentation json config used in training: "
                                         + cfg.INPUT.ALBUMENTATIONS_JSON)
        self.aug = self._get_aug(cfg.INPUT.ALBUMENTATIONS_JSON)
        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # will be modified by code below
        image = utils.read_image(dataset_dict['file_name'], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if 'annotations' not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            bboxes = [ann['bbox'] for ann in dataset_dict['annotations']]
            labels = [ann['category_id'] for ann in dataset_dict['annotations']]

            augm_annotation = self.aug(image=image, bboxes=bboxes, category_id=labels)
            image = augm_annotation['image']
            h, w = image.shape[:2]

            augm_boxes = np.array(augm_annotation['bboxes'], dtype=np.float32)
            # sometimes bbox annotations go beyond image
            augm_boxes[:, :] = augm_boxes[:, :].clip(min=[0,0,0,0], max=[w,h,w,h])

            # TODO BUG augm_boexes does not equalt to bboxes, when bboxes lost, augm_boxes becomre []
            # might be the ShiftScaleRotate bug ?
            # try:
            #     augm_boxes[:, :] = augm_boxes[:, :].clip(min=[0,0,0,0], max=[w,h,w,h])
            # except:
            #     print(augm_boxes, augm_annotation)
            #     print(dataset_dict['file_name'])
            #     print(dataset_dict)
            #     print(bboxes)

            augm_labels = np.array(augm_annotation['category_id'])

            try:
                box_mode = dataset_dict['annotations'][0]['bbox_mode']
            except:
                raise AttributeError('line 162 in dataset_mapper.py failed, please check your dataset/dataset_dict')

            dataset_dict['annotations'] = [
                {
                    'iscrowd': 0,
                    'bbox': augm_boxes[i].tolist(),
                    'category_id': augm_labels[i],
                    'bbox_mode': box_mode
                }
                for i in range(len(augm_boxes))
            ]
            if self.crop_gen:
                # detecton2 CROP, Generate a CropTransform so that the cropping region contains
                # the center of the given instance.
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"])  # TODO BUG: this line is buggy, choice from []
                )
                image = crop_tfm.apply_image(image)
            # detectron2 Resize transforms
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1).astype('float32'))
        )
        dataset_dict['height'] = image.shape[0]
        dataset_dict['width'] = image.shape[1]

        if not self.is_train:
            dataset_dict.pop('annotations', None)
            dataset_dict.pop('sem_seg_file_name', None)
            return dataset_dict

        if 'annotations' in dataset_dict:
            # we do not care about segmentation and keypoints
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image.shape[:2]
                )
                for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0
            ]
            # convert annotations to Instances to be used by detectron2 models
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict['instances'] = utils.filter_empty_instances(instances)
        # import pdb; pdb.set_trace
        # import pickle
        # with open('/data/tmp/album_dataset_dict.pkl', 'wb') as f:
        #     pickle.dump(dataset_dict, f)
        return dataset_dict

    def _get_aug(self, arg):
        with open(arg) as f:
            return A.from_dict(json.load(f))