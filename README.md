# Imitated Detector

This project is based on mmdetection(v-2.9.0), all the usage is the same as [mmdetection](https://mmdetection.readthedocs.io/en/latest/) including training , test and so on.

# Installation

* Set up a new conda environment: `conda create -n distiller python=3.7`

* Install pytorch

* [Install mmcv ( 1.2.4 <= mmcv-full < 1.3 )](https://github.com/open-mmlab/mmcv#installation)

* Install mmdetection-distiller

  ```bash
  git clone https://github.com/pppppM/mmdetection-distiller.git
  cd mmdetection-distiller
  pip install -r requirements/build.txt
  pip install -v -e .
  ```

# Dataset
* Set up MS COCO detasets, VOC2007  and VOC2012
* Make the PDD dataset  
```
python dataset/generate_pdd_dataset.py
``` 
* Make the PDD+ODD dataset  
```
python dataset/generate_pdd_odd_dataset.py
``` 
* Generate the synthetic images   
```
python dataset/img_synthetic.py
```
* Generate the adversarial images   
```
python dataset/transfer_attack/mtidifgsm_for_detection_adv_aug.py
```
* Copy the annotation file to the image dataset   
```
mv anntations/* to /ROOT/data
```

# Train Teacher

```
#single GPU
bash ./tools/dist_train.sh configs/retinanet/retinanet_x101_64x4d_fpn_1x_voc_0712.py 1 --cfg-option=work_dir=/results/model_stealing/result/teacher_voc

#multi GPU
bash ./tools/dist_train.sh configs/retinanet/retinanet_x101_64x4d_fpn_1x_voc_0712.py 8 --cfg-option=work_dir=/results/model_stealing/result/teacher_voc
```

# Generate Persudo Label
```
bash ./tools/dist_test.sh configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco_20.py /root/results/model_stealing/result/teacher_voc/latest.pth 8 --gen_psd_label --show --cfg-option data.test.keep_ratio=25 data.train.ann_file=/ROOT/data/VOCdevkit/MTIDIFGSM_1_coco20_train.json
```
# Train Imitated Detector by a Two-stage Training Strategy
```
# one stage
bash ./tools/dist_train.sh configs/fsaf/fsaf_r50_fpn_2x_coco_20.py 8 --cfg-option work_dir=/ROOT/results/model_stealing/result/fsaf_MDIFGSM_0_4_img_syn_train_correct data.train.ann_file=/ROOT/data/VOCdevkit/psd_MDIFGSM_4_coco20_train_correct_psd_MTIDIFGSM_0_coco20_train_correct_psd_img_synthesis.json
```

```
# two stage
bash ./tools/dist_train.sh configs/fsaf/fsaf_r50_fpn_2x_coco_20.py  8 --cfg-option work_dir=/ROOT/result/fasf_MDIFGSM_0_4_img_syn_train_correct_resume12_coco20_train_161822 data.train.ann_file=/ROOT/data/VOCdevkit/psd_coco20_train.json --resume-from=/ROOT/result/fsaf_MDIFGSM_0_4_img_syn_train_correct/epoch_12.pth
```

# Results
We show the training logs for imitated detectors with different architectures in ```results/```:

* Retinanet: 

* FCOS

* Anchor Free

* Guided Anchoring

* FASF




# Visual Analysis
You can use code ```tools/analysis_tools/analyze_logs.py``` and ```tools/analysis_tools/analyze_results.py``` to analyze the training process and the final detection result of the imitated detector .

# Thanks
* [1] https://github.com/kakaobrain/minDALL-E
* [2] https://github.com/open-mmlab/mmdetection
* [3] https://github.com/pppppM/mmdetection-distiller
