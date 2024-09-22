#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   iou_nes_attack_demo.py
@Time    :   2021/01/12 18:57:09
@Author  :   Anoymous
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import time
import sys
import os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import torch
import argparse
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import warnings
import cv2
# import utils
import time
import attack_demo.util as demo_utils
from datetime import datetime
import blackbox_attack.square_attack.utils as sq_utils
from blackbox_attack.utils.criterion import loss_fct, early_stop_crit_fct, early_stop_crit_fct_with_iou, loss_fct_with_iou
import mmcv
import os.path as osp
import sys
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import matplotlib as mpl
from res2lb import res2lb
mpl.use('Agg')
import matplotlib.pyplot as plt
from mmcv.image import tensor2imgs
from mmdet_v2200.apis import multi_gpu_test, single_gpu_test
from mmdet_v2200.core import encode_mask_results
from mmdet_v2200.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet_v2200.models import build_detector
# from blackbox_attack.nes_attack import NESAttack
from attack_demo.util import Logger, get_scores_and_labels
from transfer_attack.attacks.tifgsm import TIFGSM

def parse_args():
    # detector argument
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out_log', type=str, help='save cmd output as log file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--out_dir', default='/srv/hdd/results/od-black/mmdetection/sq_attack/det', help='directory where painted images will be saved')
    parser.add_argument(
        '--vis_step_out_dir', default='/srv/hdd/results/od-black/mmdetection/sq_attack/det', help='directory where painted images will be saved')
    parser.add_argument(
        '--score_thr',default=0.3,type=float,help='score thread used to filter bboxes with low score'
    )       
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--gen_adv_aug',action='store_true',help='test on training set and generate adv examples'
    )

    # attack argument
    parser.add_argument('--attack', type=str)
    parser.add_argument('--exp_folder', type=str, default='/srv/hdd/results/od-black/mmdetection/sign_attack/adv', help='Experiment folder to store all output.')
    parser.add_argument('--n_iter', type=str, default='10000')
    parser.add_argument('--p', type=str, default='inf', choices=['inf', 'l2'])
    parser.add_argument('--targeted', default=False)
    parser.add_argument('--model', default='Faster-RCNN')
    parser.add_argument('--eps', type=float, default=0.05, help='Radius of the Lp ball.0.05*4.52=0.22')
    parser.add_argument('--loss', type=str, default='cw_loss')
    parser.add_argument('--fd_eta', type=float, default=0.01)
    parser.add_argument('--attack_logistics', default=None)
    parser.add_argument('--vis_attack_step', default=None)
    parser.add_argument('--zeta', type=float, default='0.5')
    parser.add_argument('--lambda1', type=float, default='1.0')
    parser.add_argument('--patch_attack', default=None)
    parser.add_argument('--lr', default=0.005)
    parser.add_argument('--q', default=50)
    args = parser.parse_args()
    # args.loss = 'margin_loss' if not args.targeted else 'cross_entropy'
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def main():
    args = parse_args()
    if args.attack_logistics is not None:
        attack_logistics = bool(args.attack_logistics)
    else:
        attack_logistics = None

    if args.vis_attack_step is not None:
            vis_attack_step = bool(args.vis_attack_step)
    else:
        vis_attack_step = None
    
    args.zeta = float(args.zeta)
    args.lambda1 = float(args.lambda1)

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    if args.out_log is not None:
        out_log_dic = args.out_log.rsplit("/",1)[0]
        if not os.path.exists(out_log_dic):
            os.makedirs(out_log_dic)
        if os.path.exists(args.out_log):
            os.remove(args.out_log)
        log = Logger(args.out_log, level='info')
    
    log.logger.info(args)
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    if args.gen_adv_aug:
        cfg.data.test.ann_file = cfg.data.test.ann_file
        cfg.data.test.img_prefix = cfg.data.test.img_prefix
        cfg.data.val.ann_file = cfg.data.test.ann_file
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    # dataset = build_dataset(cfg.data.test)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        # workers_per_gpu=0,
        dist=distributed,
        shuffle=False)
    log.logger.info('Load data from the path:{}'.format(cfg.data.test['img_prefix']))

    # build the model and load checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    log.logger.info('checkpoints have been finishe into object detection model...')

    # initilaze object detector
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # prepare dataset (test dataset: 991 images)
    dataset = data_loader.dataset

    timestamp = str(datetime.now())[:-7]

    # init log file

    
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    # results_records = [ [] for _ in range(len(results_records_iter_list))]
    # quires_records = [ [] for _ in range(len(results_records_iter_list))]
    results = []
    for index, data in enumerate(data_loader):
    #     continue
        # if i != 838:
        #     continue
        # get the images bound and ori_img
        x_numpy = data['img'][0].numpy().transpose(0, 2, 3, 1)
        # ori_img = torch.FloatTensor(x_numpy.copy().transpose(0, 3, 1, 2))
        ori_img = torch.FloatTensor(x_numpy.transpose(0, 3, 1, 2)).cuda()
        lb, ub = x_numpy.min(), x_numpy.max()
        with torch.no_grad():
        #     # get logistic scores
            result = model(return_loss=False, rescale=False, attack_mode=True, attack_logistics=attack_logistics, **data)
            gt_bboxes, _, gt_labels = demo_utils.get_bboxes_scores_and_labels(result, ncls=len(dataset.CLASSES), to_Tensor=True)

        
        # if attack_logistics is not None and attack_logistics:
        #     with torch.no_grad():
        #     # get logistic scores ````````````
        #         result = model(return_loss=False, rescale=True, attack_mode=True, attack_logistics=attack_logistics, **data)
        #     _, labels_clean = demo_utils.get_scores_and_labels(result, ncls=len(dataset.CLASSES))
        #     scores_logit_clean = result[2][0].cpu().detach().numpy()
        # else:
        #     with torch.no_grad():
        #     # get the bbox_scores:[n], bbox_lables[n] 
        #         result = model(return_loss=False, rescale=True, attack_mode=True, **data)
        #     # get softmax scores and labels
        #     scores_smooth_clean, labels_clean = demo_utils.get_scores_and_labels(result, ncls=len(dataset.CLASSES))
   
        # define the targeted label
        # labels_target  = sq_utils.random_classes_except_current(labels_clean, len(dataset.CLASSES)) if args.targeted else labels_clean
        # labels_target_onehot = sq_utils.dense_to_onehot(labels_target, n_cls=len(dataset.CLASSES))

        # define the attacker
        # TIFGSM
        # attack = TIFGSM(model, eps=12.5/255, alpha=3/255, steps=int(args.n_iter), decay=0.0, resize_rate=0.9, diversity_prob=0.0, random_start=False, ub=ub, lb=lb)
        
        # M-TIFGSM
        # attack = TIFGSM(model, eps=12.5/255, alpha=3/255, steps=int(args.n_iter), decay=1.0, resize_rate=0.9, diversity_prob=0.0, random_start=False, ub=ub, lb=lb)
        
        # TI-DI-FGSM
        # attack = TIFGSM(model, eps=12.5/255, alpha=3/255, steps=int(args.n_iter), decay=0.0, resize_rate=0.9, diversity_prob=0.7, random_start=False, ub=ub, lb=lb)
        
        # M-TI-DI-FGSM
        attack = TIFGSM(model, eps=12.5/255, alpha=3/255, steps=int(args.n_iter), decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=True, ub=ub, lb=lb)

        adv = attack(data, (gt_bboxes, gt_labels))
        time_start = time.time()

        # attack optimize
        # if vis_attack_step is not None and vis_attack_step:
        #     queries, vis_result, results_records, queries_records = attacker.run(data, loss_fct_with_iou, early_stop_crit_fct_with_iou, vis_attack_step=vis_attack_step, results_records=results_records, quires_records=quires_records)
        #     # queries, vis_result = attacker.run(data, loss_fct_with_iou, early_stop_crit_fct_with_iou, vis_attack_step=vis_attack_step, log=log)
        # else:
        #     queries = attacker.run(data, loss_fct, early_stop_crit_fct)
        # queries = [0.0]

        # save the adversarial example
        if len(adv) == 2 and args.exp_folder:
            for i in range(len(adv)):
                data['img'][0] = adv[i]
                if args.exp_folder:
                    img_tensor = data['img'][0]
                    img_metas = data['img_metas'][0].data[0]
                    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                    assert len(imgs) == len(img_metas)

                    for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                        if args.model == "CornerNet":
                            h, w, _ = img_meta['pad_shape']
                        else:
                            h, w, _ = img_meta['img_shape']
                        img_show = img[:h, :w, :]

                        ori_h, ori_w = img_meta['ori_shape'][:-1]
                        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                        attack_method = args.attack + '_' + str(args.n_iter)
                        out_path = osp.join(args.exp_folder, attack_method, 'JPEGImages')

                        if not os.path.exists(out_path):
                            os.makedirs(out_path)

                        if args.exp_folder:
                            out_file = osp.join(out_path, img_meta['filename'].split('/')[-1])
                        else:
                            out_file = None

                        cv2.imwrite(out_file, img_show)
        else:
            data['img'][0] = adv
            if args.exp_folder:
                img_tensor = data['img'][0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    if args.model == "CornerNet":
                        h, w, _ = img_meta['pad_shape']
                    else:
                        h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    attack_method = args.attack + '_' + str(args.n_iter)
                    out_path = osp.join(args.exp_folder, attack_method, 'JPEGImages')

                    if not os.path.exists(out_path):
                        os.makedirs(out_path)

                    if args.exp_folder:
                        out_file = osp.join(out_path, img_meta['filename'].split('/')[-1])
                    else:
                        out_file = None

                    cv2.imwrite(out_file, img_show)

        # save the result on the adversarial example
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
            # batch_size = len(result)
            # out_dir = args.out_dir
            # if out_dir:
            #     if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
            #         img_tensor = data['img'][0]
            #     else:
            #         img_tensor = data['img'][0].data[0]
            #     img_metas = data['img_metas'][0].data[0]
            #     imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            #     assert len(imgs) == len(img_metas)

            #     for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
            #         if args.model == "CornerNet":
            #             h, w, _ = img_meta['pad_shape']
            #         else:
            #             h, w, _ = img_meta['img_shape']
            #         img_show = img[:h, :w, :]

            #         ori_h, ori_w = img_meta['ori_shape'][:-1]
            #         img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            #         if out_dir:
            #             out_file = osp.join(out_dir, args.model, img_meta['filename'].split('/')[-1])
            #         else:
            #             out_file = None
                    # cv2.imwrite(out_file, img_show)
                    # model.module.show_result(
                    #     img_show,
                    #     result[i],
                    #     show=True,
                    #     out_file=out_file,
                    #     score_thr=0.7)
        
        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(1):
            prog_bar.update()

        time_total = time.time() - time_start
        # log.logger.info('IMAGE {}: quries {}, times {}'.format(index, queries, time_total))
        # total_quires.append(queries)
        # total_times.append(time_total)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            # print('writing results to {args.out}')
            log.logger.info('writing results to {}'.format(args.out))
            mmcv.dump(results, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        
        if args.gen_adv_aug  and cfg.data.test.type=='CocoDataset':
            result_f = dataset.result2jsonF(results)
            test_ann_file = cfg.data.test.ann_file
            label_json = res2lb(result_f,json.load(open(test_ann_file,'r')),score_thr=args.score_thr)
            json_str = json.dumps(label_json)
            json_str = json_str.replace("VOC2007", attack_method)
            label_out = test_ann_file.replace(test_ann_file.split('/')[-1], attack_method + '_'+test_ann_file.split('/')[-1])
            with open(label_out, 'w') as json_file:
                json_file.write(json_str)

        if args.format_only:
            dataset.format_results(results, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(results, **eval_kwargs))

            # visualize the precision_recall image, IoU=0.5
            # eval_kwargs['precision_recall_visual'] = True
            # eval_results, precision = dataset.evaluate(results, **eval_kwargs)
            # x = np.arange(0.0, 1.01, 0.01)
            # plt.xlabel('recall')
            # plt.ylabel('precision')
            # plt.xlim(0, 1.0)
            # plt.ylim(0, 1.01)
            # plt.grid(True)
            # plt.plot(x, precision, 'b-', label='YOLOv3')
            # # plt.show()
            # plt.savefig('test.jpg')
            # log.logger.info(eval_results)


if __name__ == "__main__":
    main()
