import json

def res2lb(res,ori_lb,score_thr):
    psd_ann = []
    img_id_list = []
    ann_id  = 0
    for ann in res:
        if ann['score'] > score_thr:
            annotation = dict()
            annotation["image_id"] = ann['image_id']
            annotation["segmentation"] = []
            annotation["bbox"] = [int(i) for i in ann['bbox']]
            annotation["category_id"] = ann['category_id']
            annotation["id"] = ann_id
            annotation["iscrowd"] = 0
            annotation["area"] =annotation["bbox"][2] * annotation["bbox"][3]
            annotation["ignore"] = 0
            psd_ann.append(annotation)
            ann_id+=1

            if ann['image_id'] not in img_id_list:
                img_id_list.append(ann['image_id'])

    psd_coco = dict()
    psd_coco['images'] =[img for img in ori_lb['images'] if img['id'] in img_id_list]
    psd_coco['type'] = ori_lb['type']
    psd_coco['categories'] = ori_lb['categories']
    psd_coco['annotations'] = psd_ann
    return psd_coco



if __name__ == '__main__':
    ori_lb_path = '/home/jwl/code/MS_voc/data/VOCdevkit/anno_coco/voc07_trainval.json'
    res_path = '/home/jwl/code/MS_voc/result.bbox.json'
    psd_lb_path = '/home/jwl/code/MS_voc/data/VOCdevkit/anno_coco/psd_train.json'
    score_thr = 0.3

    res = json.load(open(res_path,'r'))
    ori_lb = json.load(open(ori_lb_path,'r'))

    # psd_ann = []
    # ann_id  = 0
    # for ann in res:
    #     if ann['score'] > score_thr:
    #         annotation = dict()
    #         annotation["image_id"] = ann['image_id']
    #         annotation["segmentation"] = []
    #         annotation["bbox"] = [int(i) for i in ann['bbox']]
    #         annotation["category_id"] = ann['category_id']
    #         annotation["id"] = ann_id
    #         annotation["iscrowd"] = 0
    #         annotation["area"] =annotation["bbox"][2] * annotation["bbox"][3]
    #         annotation["ignore"] = 0
    #         psd_ann.append(annotation)
    #         ann_id+=1
    #
    # new_coco = dict()
    # new_coco['images'] = ori_lb['images']
    # new_coco['type'] = ori_lb['type']
    # new_coco['categories'] = ori_lb['categories']
    # new_coco['annotations'] = psd_ann
    psd_coco = res2lb(res,ori_lb,score_thr)
    json_str = json.dumps(psd_coco)
    with open(psd_lb_path, 'w') as json_file:
        json_file.write(json_str)

    pass
